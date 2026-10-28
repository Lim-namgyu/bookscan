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
class BookRegion(modelBytes: ByteArray, threads: Int = 4) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        modelBytes,
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(threads)
        }
    )
    private val inputName: String = session.inputNames.first()
    private val outputName: String = session.outputNames.first() // 합쳐진 최종 지도(나머지 6개는 중간 단계)

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
            session.run(mapOf(inputName to t), setOf(outputName)).use { res ->
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
    /** 마지막 mask() 호출이 null 을 돌려준 이유 (촬영 안내에 사용) */
    @Volatile var lastReason: Int = REASON_OK
        private set

    fun mask(rgb: Mat): Pair<Mat, Rect>? {
        lastReason = REASON_OK
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
            lbl.release(); st.release(); lastReason = REASON_NONE; return null
        }
        val stats = IntArray(n * 5).also { st.get(0, 0, it) }
        st.release()
        var big = 0
        for (i in 1 until n) big = max(big, stats[i * 5 + Imgproc.CC_STAT_AREA])
        if (big.toDouble() / (SIZE * SIZE) < 0.08) {
            lbl.release(); lastReason = REASON_SMALL; return null
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
        if (paperOutside(rgb, comp) > PAPER_OUTSIDE_MAX) {
            // 책 바깥이어야 할 곳이 대부분 종이(밝고 매끈함) → 페이지가 화면을 꽉 채웠거나 표·그림만 잡은 것.
            // 이런 마스크로 자르면 페이지 일부가 잘려 나가므로 쓰지 않는다.
            comp.release()
            lastReason = REASON_FILLS
            return null
        }
        val m = Mat()
        Imgproc.resize(comp, m, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        comp.release()
        Imgproc.threshold(m, m, 127.0, 255.0, Imgproc.THRESH_BINARY)
        val box = Imgproc.boundingRect(m)
        return Pair(m, box)
    }

    /** region.paper_outside: 마스크(SIZE 격자, 0/255) 바깥 영역 중 종이처럼 보이는(밝고 매끈한) 비율. 긴 변 500px 에서 잰다 */
    private fun paperOutside(rgb: Mat, comp: Mat): Double {
        val h = rgb.rows()
        val w = rgb.cols()
        val s = 500.0 / max(h, w)
        val sw = max(1, Math.rint(w * s).toInt())
        val sh = max(1, Math.rint(h * s).toInt())
        val small = Mat()
        Imgproc.resize(rgb, small, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val g8 = Mat()
        Imgproc.cvtColor(small, g8, Imgproc.COLOR_RGB2GRAY)
        small.release()
        val mm = Mat()
        Imgproc.resize(comp, mm, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val gF = Mat()
        g8.convertTo(gF, CvType.CV_32F)
        val gx = Mat()
        val gy = Mat()
        Imgproc.Sobel(gF, gx, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(gF, gy, CvType.CV_32F, 0, 1, 3)
        val n = sw * sh
        val ax = FloatArray(n).also { gx.get(0, 0, it) }
        val ay = FloatArray(n).also { gy.get(0, 0, it) }
        val sm = ByteArray(n) { if (ax[it] * ax[it] + ay[it] * ay[it] < 1600f) 1 else 0 }
        val smM = Mat(sh, sw, CvType.CV_8UC1).also { it.put(0, 0, sm) }
        val k5 = Mat.ones(5, 5, CvType.CV_8UC1)
        Imgproc.erode(smM, smM, k5)
        k5.release()
        smM.get(0, 0, sm)
        val mb = ByteArray(n).also { mm.get(0, 0, it) }
        val inside = BooleanArray(n) { (mb[it].toInt() and 0xFF) > 127 }
        val outM = Mat(sh, sw, CvType.CV_8UC1).also { it.put(0, 0, ByteArray(n) { i -> if (inside[i]) 1 else 0 }) }
        val k15 = Mat.ones(15, 15, CvType.CV_8UC1)
        Imgproc.dilate(outM, outM, k15)
        k15.release()
        val dil = ByteArray(n).also { outM.get(0, 0, it) }
        val gb = ByteArray(n).also { g8.get(0, 0, it) }
        listOf(g8, mm, gF, gx, gy, smM, outM).forEach { it.release() }
        val insideVals = ArrayList<Double>()
        for (i in 0 until n) if (inside[i]) insideVals.add((gb[i].toInt() and 0xFF).toDouble())
        if (insideVals.isEmpty()) return 0.0
        val paper = BookScanEngine.percentile(insideVals.toDoubleArray(), 75.0)
        var nOut = 0
        var nPaper = 0
        for (i in 0 until n) {
            if (dil[i].toInt() != 0) continue
            nOut++
            if ((gb[i].toInt() and 0xFF) >= 0.85 * paper && sm[i].toInt() != 0) nPaper++
        }
        return if (nOut == 0) 0.0 else nPaper.toDouble() / nOut
    }

    companion object {
        const val SIZE = 320

        const val REASON_OK = 0
        const val REASON_NONE = 1   // 두드러진 물체가 없음
        const val REASON_SMALL = 2  // 책이 너무 작게 잡힘(화면의 8% 미만)
        const val REASON_FILLS = 3  // 종이가 화면 밖까지 이어짐(책이 잘렸거나 화면을 꽉 채움)

        /** 마스크 바깥 중 종이처럼 보이는 비율이 이보다 크면 마스크를 믿지 않음 */
        const val PAPER_OUTSIDE_MAX = 0.15
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
