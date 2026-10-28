package com.bookscan.app.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * 책 영역 찾기 (engine/region.py 와 같은 처리)
 *   U^2-Net-p(두드러진 물체 분할, Apache-2.0)로 사진에서 책(펼친 페이지 전체)을 바닥·책상과 분리한다.
 *   - 책이 작게 찍혀도 책 부분만 잘라 다음 단계에 넘기고
 *   - 곡면 편 결과에서 책 바깥(바닥 무늬)을 지우는 데 쓴다.
 * 입력은 RGB.
 */
class BookRegion(modelBytes: ByteArray) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        modelBytes,
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(4)
        }
    )
    private val inputName: String = session.inputNames.first()

    override fun close() {
        session.close()
    }

    /** 0~1 확률 지도 (SIZE x SIZE, 행 우선) */
    private fun saliency(rgb: Mat): FloatArray {
        val small = Mat()
        Imgproc.resize(rgb, small, Size(SIZE.toDouble(), SIZE.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val plane = SIZE * SIZE
        val bytes = ByteArray(plane * 3)
        small.get(0, 0, bytes)
        small.release()
        var mx = 1e-6f
        for (b in bytes) mx = max(mx, (b.toInt() and 0xFF).toFloat())
        val arr = FloatArray(plane * 3)
        for (i in 0 until plane) {
            for (c in 0 until 3) {
                val v = (bytes[i * 3 + c].toInt() and 0xFF) / mx
                arr[c * plane + i] = (v - MEAN[c]) / STD[c]
            }
        }
        val out = FloatArray(plane)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { t ->
            session.run(mapOf(inputName to t)).use { res ->
                (res.get(0) as OnnxTensor).floatBuffer.get(out)
            }
        }
        var lo = Float.MAX_VALUE
        var hi = -Float.MAX_VALUE
        for (v in out) {
            lo = min(lo, v); hi = max(hi, v)
        }
        val d = max(hi - lo, 1e-6f)
        for (i in out.indices) out[i] = (out[i] - lo) / d
        return out
    }

    /**
     * 책 영역 마스크(원본 크기 8UC1, 0/255)와 경계 상자. 못 찾으면 null.
     * 가장 큰 덩어리 + 그만큼 큰 덩어리들(접힌 곳이 어두워 양쪽 페이지가 따로 잡히는 경우), 안쪽 구멍은 메움.
     */
    fun mask(rgb: Mat): Pair<Mat, Rect>? {
        val w = rgb.cols()
        val h = rgb.rows()
        val p = saliency(rgb)
        val b = ByteArray(SIZE * SIZE) { if (p[it] > 0.5f) 1 else 0 }
        val bm = Mat(SIZE, SIZE, CvType.CV_8UC1).also { it.put(0, 0, b) }
        val lbl = Mat()
        val st = Mat()
        val cen = Mat()
        val n = Imgproc.connectedComponentsWithStats(bm, lbl, st, cen, 8, CvType.CV_32S)
        bm.release(); cen.release()
        if (n < 2) {
            lbl.release(); st.release(); return null
        }
        val stats = IntArray(n * 5).also { st.get(0, 0, it) }
        st.release()
        var big = 0
        for (i in 1 until n) big = max(big, stats[i * 5 + Imgproc.CC_STAT_AREA])
        if (big.toDouble() / (SIZE * SIZE) < 0.08) {
            lbl.release(); return null
        }
        val keep = BooleanArray(n) { it > 0 && stats[it * 5 + Imgproc.CC_STAT_AREA] >= big * 0.2 }
        val labels = IntArray(SIZE * SIZE).also { lbl.get(0, 0, it) }
        lbl.release()
        val comp = Mat(SIZE, SIZE, CvType.CV_8UC1).also { m ->
            m.put(0, 0, ByteArray(SIZE * SIZE) { if (keep[labels[it]]) 255.toByte() else 0 })
        }
        // 구멍 메우기
        val cnts = ArrayList<MatOfPoint>()
        val hier = Mat()
        Imgproc.findContours(comp, cnts, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hier.release()
        comp.setTo(Scalar(0.0))
        Imgproc.drawContours(comp, cnts, -1, Scalar(255.0), -1)
        cnts.forEach { it.release() }
        val m = Mat()
        Imgproc.resize(comp, m, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        comp.release()
        Imgproc.threshold(m, m, 127.0, 255.0, Imgproc.THRESH_BINARY)
        val box = Imgproc.boundingRect(m)
        return Pair(m, box)
    }

    companion object {
        const val SIZE = 320
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        /**
         * 가장 큰 덩어리의 볼록 껍질(0/255). 한 쪽 페이지는 거의 볼록하므로 가장자리가 패인 곳을 메운다.
         * (양면 전체에 쓰면 접힌 곳 위아래의 바닥까지 덮으므로 페이지마다 따로 쓴다)
         */
        fun hullMask(m: Mat?): Mat? {
            if (m == null || Core.countNonZero(m) == 0) return null
            val lbl = Mat()
            val st = Mat()
            val cen = Mat()
            val bin = Mat()
            Imgproc.threshold(m, bin, 127.0, 255.0, Imgproc.THRESH_BINARY)
            val n = Imgproc.connectedComponentsWithStats(bin, lbl, st, cen, 8, CvType.CV_32S)
            cen.release()
            val stats = IntArray(n * 5).also { st.get(0, 0, it) }
            st.release()
            var bi = 1
            for (i in 1 until n) if (stats[i * 5 + Imgproc.CC_STAT_AREA] > stats[bi * 5 + Imgproc.CC_STAT_AREA]) bi = i
            val one = Mat()
            Core.compare(lbl, Scalar(bi.toDouble()), one, Core.CMP_EQ)
            lbl.release(); bin.release()
            val cnts = ArrayList<MatOfPoint>()
            val hier = Mat()
            Imgproc.findContours(one, cnts, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            hier.release(); one.release()
            if (cnts.isEmpty()) return null
            val c = cnts.maxByOrNull { Imgproc.contourArea(it) }!!
            val hullIdx = MatOfInt()
            Imgproc.convexHull(c, hullIdx)
            val pts = c.toArray()
            val hp = MatOfPoint(*hullIdx.toArray().map { pts[it] }.toTypedArray())
            hullIdx.release()
            val out = Mat.zeros(m.rows(), m.cols(), CvType.CV_8UC1)
            Imgproc.fillConvexPoly(out, hp, Scalar(255.0))
            hp.release()
            cnts.forEach { it.release() }
            return out
        }

        /** 경계 상자를 여유 있게 넓혀 사진 안으로 자른 상자. 거의 전체면 null */
        fun cropBox(box: Rect, w: Int, h: Int, margin: Double = 0.04): Rect? {
            val mx = box.width * margin
            val my = box.height * margin
            val x0 = max(0, (box.x - mx).toInt())
            val y0 = max(0, (box.y - my).toInt())
            val x1 = min(w, ceil(box.x + box.width + mx).toInt())
            val y1 = min(h, ceil(box.y + box.height + my).toInt())
            if ((x1 - x0).toLong() * (y1 - y0) > 0.92 * w * h) return null
            return Rect(x0, y0, x1 - x0, y1 - y0)
        }
    }
}
