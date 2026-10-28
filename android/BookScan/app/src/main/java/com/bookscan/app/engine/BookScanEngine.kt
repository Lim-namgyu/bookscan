package com.bookscan.app.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

enum class Layout { AUTO, SINGLE, DOUBLE }
enum class ColorMode { COLOR, GRAY, BW }
enum class Side { LEFT, RIGHT }

/** 곡면을 편 페이지 한 장. flat 은 RGB 8UC3. skipSide 는 손가락 검사에서 뺄 쪽(접힌 선 쪽). */
class DewarpedPage(val flat: Mat, val skipSide: Side?)

/**
 * 책 스캔 보정 엔진 (파이썬 프로토타입 engine/bookscan.py 와 같은 순서·같은 수치)
 *  1) 양면 펼침 감지 → 접힌 선 기준 좌/우 분리
 *  2) UVDoc(ONNX) 곡면 펴기 + 실제 페이지 비율 복원
 *  3) 손가락 감지·제거 (종이색 대비 피부색 + 가장자리 연결 + 두께 조건)
 *  4) 조명·그림자 평탄화, 대비·선명도 (컬러 / 회색 / 흑백)
 * 모든 Mat 은 RGB 순서를 쓴다.
 */
class BookScanEngine(modelBytes: ByteArray) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        modelBytes,
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(4)
        }
    )

    override fun close() {
        session.close()
    }

    // ================================================================= 공개 API

    /** 1단계: 사진(RGB) → 곡면 편 페이지 목록 (양면이면 2장) */
    fun dewarpPages(rgb: Mat, layout: Layout): List<DewarpedPage> {
        // 폰을 세로로 든 채 펼친 책을 찍으면 글이 옆으로 누워 있음 → 글이 바로 서도록 먼저 돌림
        val rot = PageAnalysis.detectRotation(rgb)
        val src = if (rot != 0) PageAnalysis.rotate(rgb, rot) else rgb
        try {
            val gutter = if (layout == Layout.SINGLE) null else PageAnalysis.findGutter(src, layout == Layout.DOUBLE)
            val parts = if (gutter != null) splitSpread(src, gutter) else listOf(src)
            return parts.mapIndexed { i, p ->
                val flat = rectify(p)
                if (p !== src) p.release()
                val skip = if (parts.size == 1) null else if (i == 0) Side.RIGHT else Side.LEFT
                DewarpedPage(flat, skip)
            }
        } finally {
            if (src !== rgb) src.release()
        }
    }

    /** 2단계: 손가락 제거 + 화질 보정. 결과는 RGB 8UC3(컬러) 또는 8UC1(회색·흑백) */
    fun clean(flat: Mat, mode: ColorMode, removeFingers: Boolean, skipSide: Side?): Mat {
        val mask = if (removeFingers) fingerMask(flat, skipSide) else null
        val hasMask = mask != null && Core.countNonZero(mask) > 0
        val filled = if (hasMask) fillPaper(flat, mask!!) else flat
        val norm = flattenIllumination(filled)
        if (filled !== flat) filled.release()
        val out = finish(norm, mode)
        norm.release()
        if (hasMask) erase(out, mask!!)
        mask?.release()
        cleanEdges(out)
        return out
    }

    // ================================================================= 1. 양면 분리

    /** 접힌 선 기준 좌/우 페이지. 반대쪽은 배경색으로 가린다. */
    private fun splitSpread(img: Mat, gutter: Pair<Double, Double>): List<Mat> {
        val w = img.cols()
        val h = img.rows()
        val (a, b) = gutter
        val xTop = b
        val xBot = a * (h - 1) + b
        val bg = borderMedianColor(img)
        val result = ArrayList<Mat>()
        for (left in listOf(true, false)) {
            val out = img.clone()
            val poly = if (left) {
                MatOfPoint(Point(xTop, 0.0), Point(w.toDouble(), 0.0), Point(w.toDouble(), h.toDouble()), Point(xBot, h.toDouble()))
            } else {
                MatOfPoint(Point(0.0, 0.0), Point(xTop, 0.0), Point(xBot, h.toDouble()), Point(0.0, h.toDouble()))
            }
            Imgproc.fillPoly(out, listOf(poly), bg)
            val lo = if (left) 0 else max(0, min(xTop, xBot).toInt() - 10)
            val hi = if (left) min(w, max(xTop, xBot).toInt() + 10) else w
            val crop = out.submat(0, h, lo, hi).clone()
            out.release()
            result.add(crop)
        }
        return result
    }

    private fun borderMedianColor(img: Mat): Scalar {
        val w = img.cols()
        val rows = min(8, img.rows() / 2)
        val n = rows * w * 3
        val top = ByteArray(n)
        val bot = ByteArray(n)
        img.submat(0, rows, 0, w).clone().also { it.get(0, 0, top); it.release() }
        img.submat(img.rows() - rows, img.rows(), 0, w).clone().also { it.get(0, 0, bot); it.release() }
        val ch = Array(3) { DoubleArray(2 * rows * w) }
        var k = 0
        for (arr in listOf(top, bot)) {
            var i = 0
            while (i < arr.size) {
                ch[0][k] = (arr[i].toInt() and 0xFF).toDouble()
                ch[1][k] = (arr[i + 1].toInt() and 0xFF).toDouble()
                ch[2][k] = (arr[i + 2].toInt() and 0xFF).toDouble()
                k++; i += 3
            }
        }
        return Scalar(median(ch[0]), median(ch[1]), median(ch[2]))
    }

    // ================================================================= 2. 곡면 펴기 + 글줄 반듯하게

    /**
     * 사진 한 쪽 → 반듯한 페이지.
     *   곡면 펴기 지도(모델) ∘ 테두리 자르기 ∘ 글줄 반듯하게 지도 를 합성해서
     *   원본 사진에서 단 한 번만 재샘플링한다(화질 손실 최소).
     */
    private fun rectify(img: Mat): Mat {
        val w = img.cols()
        val h = img.rows()
        val inp = Mat()
        Imgproc.resize(img, inp, Size(MODEL_W.toDouble(), MODEL_H.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val plane = MODEL_W * MODEL_H
        val bytes = ByteArray(plane * 3)
        inp.get(0, 0, bytes)
        inp.release()
        val arr = FloatArray(plane * 3)
        for (i in 0 until plane) {
            arr[i] = (bytes[i * 3].toInt() and 0xFF) / 255f
            arr[plane + i] = (bytes[i * 3 + 1].toInt() and 0xFF) / 255f
            arr[2 * plane + i] = (bytes[i * 3 + 2].toInt() and 0xFF) / 255f
        }
        val grid = FloatArray(2 * GRID_H * GRID_W)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), longArrayOf(1, 3, MODEL_H.toLong(), MODEL_W.toLong())).use { t ->
            session.run(mapOf("image" to t)).use { res ->
                (res.get(0) as OnnxTensor).floatBuffer.get(grid)
            }
        }

        // 격자(-1~1) → 원본 픽셀 좌표
        var px = Mat(GRID_H, GRID_W, CvType.CV_32F)
        var py = Mat(GRID_H, GRID_W, CvType.CV_32F)
        val gx = FloatArray(GRID_H * GRID_W) { (grid[it] + 1f) / 2f * (w - 1) }
        val gy = FloatArray(GRID_H * GRID_W) { (grid[GRID_H * GRID_W + it] + 1f) / 2f * (h - 1) }
        px.put(0, 0, gx)
        py.put(0, 0, gy)

        // 격자 곡선 길이로 실제 페이지 가로·세로 추정 → 비율 복원, 해상도 유지
        val rowLen = DoubleArray(GRID_H) { r ->
            var sum = 0.0
            for (c in 0 until GRID_W - 1) {
                val i = r * GRID_W + c
                sum += hypot((gx[i + 1] - gx[i]).toDouble(), (gy[i + 1] - gy[i]).toDouble())
            }
            sum
        }
        val colLen = DoubleArray(GRID_W) { c ->
            var sum = 0.0
            for (r in 0 until GRID_H - 1) {
                val i = r * GRID_W + c
                sum += hypot((gx[i + GRID_W] - gx[i]).toDouble(), (gy[i + GRID_W] - gy[i]).toDouble())
            }
            sum
        }
        val ow = median(rowLen)
        var oh = median(colLen)
        val scale = min(1.15, sqrt(w.toDouble() * h / max(ow * oh, 1.0)))
        // 모델이 잡은 페이지 경계가 머리글·쪽번호에 바짝 붙는 경우가 있어, 격자를 위아래로 한 칸씩 연장.
        // 좌우는 연장하지 않는다(책 옆면 띠가 들어오면 그 위의 손가락과 엉겨 손가락 찾기가 어려워짐).
        val pxE = extendGridV(gx, GRID_EXT)
        val pyE = extendGridV(gy, GRID_EXT)
        px.release(); py.release()
        val ghE = GRID_H + 2 * GRID_EXT
        px = Mat(ghE, GRID_W, CvType.CV_32F).also { it.put(0, 0, pxE) }
        py = Mat(ghE, GRID_W, CvType.CV_32F).also { it.put(0, 0, pyE) }
        oh = oh * (ghE - 1.0) / (GRID_H - 1.0)
        val outW = max(16, rint(ow * scale))
        val outH = max(16, rint(oh * scale))

        // ---- 분석용 작은 페이지 (긴 변 ANALYSIS_SIDE)
        val a = min(1.0, ANALYSIS_SIDE / max(outW, outH).toDouble())
        val aw = max(16, rint(outW * a))
        val ah = max(16, rint(outH * a))
        val ax = upsampleGrid(px, aw, ah) // 분석 픽셀 → 원본 사진 좌표
        val ay = upsampleGrid(py, aw, ah)
        px.release(); py.release()
        val small = Mat()
        if (a < 1.0) Imgproc.resize(img, small, Size(0.0, 0.0), a, a, Imgproc.INTER_AREA) else img.copyTo(small)
        val axs = Mat()
        val ays = Mat()
        Core.multiply(ax, Scalar(a), axs)
        Core.multiply(ay, Scalar(a), ays)
        val flatA = Mat()
        Imgproc.remap(small, flatA, axs, ays, Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE, Scalar(0.0))
        small.release(); axs.release(); ays.release()

        // ---- 테두리 자르기 + 글줄 반듯하게 (분석 해상도)
        val (t, bt, l, r) = trimRect(flatA)
        val fa = flatA.submat(t, ah - bt, l, aw - r).clone()
        flatA.release()
        // 글줄 반듯하게 단계가 실패해도 페이지는 곡면 펴기 결과로 계속 진행
        val st = runCatching { Straightener.maps(fa) }.getOrNull()
        val sx: Mat
        val sy: Mat
        if (st != null) {
            sx = st.mapX; sy = st.mapY
        } else {
            val w0 = fa.cols()
            val h0 = fa.rows()
            sx = Mat(h0, w0, CvType.CV_32F); sy = Mat(h0, w0, CvType.CV_32F)
            val bx = FloatArray(w0 * h0) { (it % w0).toFloat() }
            val by = FloatArray(w0 * h0) { (it / w0).toFloat() }
            sx.put(0, 0, bx); sy.put(0, 0, by)
        }
        fa.release()
        // 반듯하게 펴기는 내용이 잘리지 않도록 캔버스를 조금 넓힐 수 있으므로 지도 크기를 기준으로
        val fw = sx.cols()
        val fh = sx.rows()
        Core.add(sx, Scalar(l.toDouble()), sx)
        Core.add(sy, Scalar(t.toDouble()), sy)

        // ---- 합성: 최종 분석 픽셀 → 분석 페이지 좌표 → 원본 사진 좌표
        val fxa = Mat()
        val fya = Mat()
        Imgproc.remap(ax, fxa, sx, sy, Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE, Scalar(0.0))
        Imgproc.remap(ay, fya, sx, sy, Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE, Scalar(0.0))
        ax.release(); ay.release(); sx.release(); sy.release()

        // ---- 원본 해상도로 확대 후 한 번만 재샘플링
        val fullW = max(16, rint(fw / a))
        val fullH = max(16, rint(fh / a))
        val fx = Mat()
        val fy = Mat()
        Imgproc.resize(fxa, fx, Size(fullW.toDouble(), fullH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        Imgproc.resize(fya, fy, Size(fullW.toDouble(), fullH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        fxa.release(); fya.release()
        val out = Mat()
        Imgproc.remap(img, out, fx, fy, Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE, Scalar(0.0))
        fx.release(); fy.release()
        val trimmed = trimDarkBorder(out)
        if (trimmed !== out) out.release()
        return trimmed
    }

    /** 격자(GRID_H x GRID_W, 행 우선)를 위아래로 n 행 직선 연장 */
    private fun extendGridV(g: FloatArray, n: Int): FloatArray {
        var rows = GRID_H
        var cur = g
        repeat(n) {
            val next = FloatArray((rows + 2) * GRID_W)
            for (c in 0 until GRID_W) {
                next[c] = 2 * cur[c] - cur[GRID_W + c]
                for (r in 0 until rows) next[(r + 1) * GRID_W + c] = cur[r * GRID_W + c]
                next[(rows + 1) * GRID_W + c] = 2 * cur[(rows - 1) * GRID_W + c] - cur[(rows - 2) * GRID_W + c]
            }
            cur = next
            rows += 2
        }
        return cur
    }

    /** align_corners 방식 업샘플: dst(x,y) = grid(x*(gw-1)/(W-1), y*(gh-1)/(H-1)) */
    private fun upsampleGrid(g: Mat, outW: Int, outH: Int): Mat {
        val m = Mat(2, 3, CvType.CV_64F)
        m.put(0, 0, (g.cols() - 1.0) / (outW - 1.0), 0.0, 0.0, 0.0, (g.rows() - 1.0) / (outH - 1.0), 0.0)
        val out = Mat()
        Imgproc.warpAffine(g, out, m, Size(outW.toDouble(), outH.toDouble()), Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP, Core.BORDER_REPLICATE)
        m.release()
        return out
    }

    private data class Trim(val top: Int, val bottom: Int, val left: Int, val right: Int)

    /** 가장자리에 남은 책상·책 옆면의 어두운 띠 두께 (위, 아래, 왼쪽, 오른쪽) */
    private fun trimRect(img: Mat, maxFrac: Double = 0.06): Trim {
        val h = img.rows()
        val w = img.cols()
        val s = min(1.0, 1000.0 / max(h, w))
        val small = Mat()
        Imgproc.resize(img, small, Size(rint(w * s).toDouble(), rint(h * s).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val gray = Mat()
        if (small.channels() == 3) Imgproc.cvtColor(small, gray, Imgproc.COLOR_RGB2GRAY) else small.copyTo(gray)
        small.release()
        val sh = gray.rows()
        val sw = gray.cols()
        val g = ByteArray(sh * sw)
        gray.get(0, 0, g)
        gray.release()
        val all = DoubleArray(g.size) { (g[it].toInt() and 0xFF).toDouble() }
        val thr = percentileByHistogram(g, 70.0) * 0.72

        fun rowMed(r: Int) = median(DoubleArray(sw) { all[r * sw + it] })
        fun colMed(c: Int) = median(DoubleArray(sh) { all[it * sw + c] })
        fun scan(limit: Int, med: (Int) -> Double): Int {
            var n = 0
            for (i in 0 until limit) {
                if (med(i) < thr) n++ else break
            }
            return n
        }
        val limH = (sh * maxFrac).toInt()
        val limW = (sw * maxFrac).toInt()
        var t = scan(limH) { rowMed(it) }
        var bt = scan(limH) { rowMed(sh - 1 - it) }
        var l = scan(limW) { colMed(it) }
        var r = scan(limW) { colMed(sw - 1 - it) }

        // 무늬 띠(바닥 카펫, 책 옆면의 종이 단면 줄무늬): 바깥에서 안쪽으로 훑다가 빈 종이 여백이 시작되는 곳까지
        val gM = Mat(sh, sw, CvType.CV_8UC1).also { it.put(0, 0, g) }
        val gxM = Mat()
        val gyM = Mat()
        Imgproc.Sobel(gM, gxM, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(gM, gyM, CvType.CV_32F, 0, 1, 3)
        gM.release()
        val gx = FloatArray(sh * sw).also { gxM.get(0, 0, it) }
        val gy = FloatArray(sh * sw).also { gyM.get(0, 0, it) }
        gxM.release(); gyM.release()
        val busyB = ByteArray(sh * sw) { if (gx[it] * gx[it] + gy[it] * gy[it] > 1600f) 1 else 0 }
        val busyM = Mat(sh, sw, CvType.CV_8UC1).also { it.put(0, 0, busyB) }
        val k5 = Mat.ones(5, 5, CvType.CV_8UC1)
        Imgproc.dilate(busyM, busyM, k5)
        k5.release()
        busyM.get(0, 0, busyB)
        busyM.release()
        val r0 = (sh * 0.1).toInt()
        val r1 = (sh * 0.9).toInt()
        val c0 = (sw * 0.1).toInt()
        val c1 = (sw * 0.9).toInt()
        val colb = DoubleArray(sw) { x ->
            var cnt = 0
            for (y in r0 until r1) if (busyB[y * sw + x].toInt() == 0) cnt++
            cnt.toDouble() / max(1, r1 - r0)
        }
        val rowb = DoubleArray(sh) { y ->
            var cnt = 0
            for (x in c0 until c1) if (busyB[y * sw + x].toInt() == 0) cnt++
            cnt.toDouble() / max(1, c1 - c0)
        }
        fun tex(v: (Int) -> Double, n: Int, limit: Int): Int {
            for (i in 0 until min(limit, n - 3)) {
                if (v(i) >= 0.8 && v(i + 1) >= 0.8 && v(i + 2) >= 0.8) {
                    if (i == 0) return 0
                    var sum = 0.0
                    for (j in 0 until i) sum += v(j)
                    return if (sum / i < 0.3) i else 0 // 잘라낼 띠가 '무늬'일 때만
                }
            }
            return 0
        }
        val limH2 = (sh * 0.10).toInt()
        val limW2 = (sw * 0.10).toInt()
        t = max(t, tex({ rowb[it] }, sh, limH2))
        bt = max(bt, tex({ rowb[sh - 1 - it] }, sh, limH2))
        l = max(l, tex({ colb[it] }, sw, limW2))
        r = max(r, tex({ colb[sw - 1 - it] }, sw, limW2))
        return Trim(rint(t / s), rint(bt / s), rint(l / s), rint(r / s))
    }

    /** 곡면 펴기 후 가장자리에 남은 책상·책 옆면의 어두운 띠를 잘라낸다. */
    private fun trimDarkBorder(img: Mat): Mat {
        val (t, bt, l, r) = trimRect(img)
        if (t + bt + l + r == 0) return img
        return img.submat(t, img.rows() - bt, l, img.cols() - r).clone()
    }

    // ================================================================= 3. 손가락 제거

    /** 종이보다 붉은(피부색) 두꺼운 덩어리 중 페이지 가장자리에서 들어온 것 → 손가락 마스크(원본 크기, 8UC1) */
    private fun fingerMask(img: Mat, skipSide: Side?): Mat {
        val h = img.rows()
        val w = img.cols()
        val s = 800.0 / max(h, w)
        val small = Mat()
        Imgproc.resize(img, small, Size((w * s).roundToInt().toDouble(), (h * s).roundToInt().toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val lab = Mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_RGB2Lab)
        small.release()
        val sh = lab.rows()
        val sw = lab.cols()
        val n = sh * sw
        val bytes = ByteArray(n * 3)
        lab.get(0, 0, bytes)
        lab.release()

        val lArr = FloatArray(n)
        val aArr = FloatArray(n)
        val bArr = FloatArray(n)
        for (i in 0 until n) {
            lArr[i] = (bytes[i * 3].toInt() and 0xFF) * 100f / 255f
            aArr[i] = (bytes[i * 3 + 1].toInt() and 0xFF) - 128f
            bArr[i] = (bytes[i * 3 + 2].toInt() and 0xFF) - 128f
        }
        val p60 = percentile(lArr, 60.0)
        val paperA = ArrayList<Float>()
        for (i in 0 until n) if (lArr[i] > p60) paperA.add(aArr[i])
        val a0 = if (paperA.isEmpty()) 0f else percentile(paperA.toFloatArray(), 50.0)

        // 피부: 종이보다 붉은기(a*)가 확실히 높고, 색상각 25~60°, 채도 있음 (바랜 종이·책 옆면은 60° 이상)
        val skinBytes = ByteArray(n)
        for (i in 0 until n) {
            val a = aArr[i]
            val b = bArr[i]
            val hue = Math.toDegrees(kotlin.math.atan2(b.toDouble(), a.toDouble()))
            val chroma = hypot(a.toDouble(), b.toDouble())
            val isSkin = (a - a0) > 8f && chroma > 14 && hue > 25 && hue < 60 && lArr[i] > 20f && lArr[i] < 90f
            skinBytes[i] = if (isSkin) 255.toByte() else 0
        }
        val skin = Mat(sh, sw, CvType.CV_8UC1)
        skin.put(0, 0, skinBytes)
        Imgproc.morphologyEx(skin, skin, Imgproc.MORPH_OPEN, ellipse(3))
        Imgproc.morphologyEx(skin, skin, Imgproc.MORPH_CLOSE, ellipse(9))

        // 두께 조건: 붉은 인쇄 글자처럼 얇은 것은 침식으로 사라진다
        val k = max(5, (min(sh, sw) * 0.025).toInt()) or 1
        val core = Mat()
        Imgproc.erode(skin, core, ellipse(k))

        val labels = Mat()
        val stats = Mat()
        val cents = Mat()
        val nComp = Imgproc.connectedComponentsWithStats(core, labels, stats, cents, 8, CvType.CV_32S)
        val mask = Mat.zeros(sh, sw, CvType.CV_8UC1)
        val edge = max(3, (min(sh, sw) * 0.02).toInt()) + k / 2
        val kern = ellipse(k + 2)
        val comp = Mat()
        for (i in 1 until nComp) {
            val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val bw = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val bh = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
            if (area < sh * sw * 0.0006) continue
            // 가장자리에 닿아야 하고(손가락은 바깥에서 들어옴), 한 변을 따라 길게 이어진 띠(책 옆면)는 제외
            // (변을 따라 길이 비율, 안쪽 깊이 비율, 변을 따라 길이 px, 깊이 px)
            val sides = ArrayList<DoubleArray>()
            if (x <= edge && skipSide != Side.LEFT) sides.add(doubleArrayOf(bh.toDouble() / sh, bw.toDouble() / sw, bh.toDouble(), bw.toDouble()))
            if (x + bw >= sw - edge && skipSide != Side.RIGHT) sides.add(doubleArrayOf(bh.toDouble() / sh, bw.toDouble() / sw, bh.toDouble(), bw.toDouble()))
            if (y <= edge) sides.add(doubleArrayOf(bw.toDouble() / sw, bh.toDouble() / sh, bw.toDouble(), bh.toDouble()))
            if (y + bh >= sh - edge) sides.add(doubleArrayOf(bw.toDouble() / sw, bh.toDouble() / sh, bw.toDouble(), bh.toDouble()))
            if (sides.isEmpty()) continue
            if (sides.any { it[0] > 0.40 }) continue
            if (sides.maxOf { it[1] } < 0.02) continue
            // 손가락은 안쪽으로 파고든다. 가장자리를 따라 가늘고 길게 붙은 띠(책 옆면·바랜 가장자리)는 제외
            if (sides.all { it[2] > 2.0 * it[3] }) continue
            // 심지를 원래 두께만큼 되살리되 피부색 영역 안에서만
            Core.compare(labels, Scalar(i.toDouble()), comp, Core.CMP_EQ)
            Imgproc.dilate(comp, comp, kern)
            Core.bitwise_and(comp, skin, comp)
            Core.bitwise_or(mask, comp, mask)
        }
        comp.release(); labels.release(); stats.release(); cents.release(); core.release(); skin.release()

        if (Core.countNonZero(mask) > 0) {
            // 손가락 테두리와 맞닿은 그림자까지 덮도록 확장
            val d = max(5, (min(sh, sw) * 0.03).toInt()) or 1
            Imgproc.dilate(mask, mask, ellipse(d))
        }
        val full = Mat()
        Imgproc.resize(mask, full, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        mask.release()
        return full
    }

    /** 손가락 자리를 주변 종이색으로 메우고, 바로 옆 그림자를 밝힌다(조명 보정 전에 해야 얼룩이 안 남음). */
    private fun fillPaper(img: Mat, mask: Mat): Mat {
        val h = img.rows()
        val w = img.cols()
        val s = 600.0 / max(h, w)
        val ss = Size((w * s).roundToInt().toDouble(), (h * s).roundToInt().toDouble())
        val small = Mat()
        Imgproc.resize(img, small, ss, 0.0, 0.0, Imgproc.INTER_AREA)
        val m = Mat()
        Imgproc.resize(mask, m, ss, 0.0, 0.0, Imgproc.INTER_NEAREST)
        val longSide = max(ss.width, ss.height)

        // 글자를 지운 '종이 배경' → 손가락 밖 종이만으로 가중 평균(정규화 컨볼루션)해 종이색 추정
        val k = max(9, (longSide * 0.03).toInt()) or 1
        val paper = Mat()
        Imgproc.morphologyEx(small, paper, Imgproc.MORPH_CLOSE, ellipse(k))
        val dil = Mat()
        Imgproc.dilate(m, dil, ellipse(5))
        val valid8 = Mat()
        Core.compare(dil, Scalar(0.0), valid8, Core.CMP_EQ)
        val valid = Mat()
        valid8.convertTo(valid, CvType.CV_32F, 1.0 / 255.0)
        val sig = longSide * 0.05
        val paperF = Mat()
        paper.convertTo(paperF, CvType.CV_32FC3)
        val num = Mat()
        Core.multiply(paperF, merge3(valid), num)
        Imgproc.GaussianBlur(num, num, Size(0.0, 0.0), sig)
        val den = Mat()
        Imgproc.GaussianBlur(valid, den, Size(0.0, 0.0), sig)
        Core.max(den, Scalar(1e-3), den)
        val ref = Mat()
        Core.divide(num, merge3(den), ref)

        // 그림자 밝히기: 글자만 지운(작은 커널) 종이 밝기를 기준 종이 밝기로 끌어올리는 배율
        val paperS = Mat()
        Imgproc.morphologyEx(small, paperS, Imgproc.MORPH_CLOSE, ellipse(7))
        val paperSF = Mat()
        paperS.convertTo(paperSF, CvType.CV_32FC3)
        val lumRef = channelMean(ref)
        val lumP = channelMean(paperSF)
        Core.max(lumP, Scalar(1.0), lumP)
        val gain = Mat()
        Core.divide(lumRef, lumP, gain)
        Core.max(gain, Scalar(1.0), gain)
        Core.min(gain, Scalar(2.5), gain)
        Imgproc.GaussianBlur(gain, gain, Size(0.0, 0.0), 1.5)
        val d = max(5, (longSide * 0.06).toInt()) or 1
        val ring8 = Mat()
        Imgproc.dilate(m, ring8, ellipse(d))
        val ringF = Mat()
        Imgproc.GaussianBlur(ring8, ringF, Size(0.0, 0.0), 3.0)
        ringF.convertTo(ringF, CvType.CV_32F, 1.0 / 255.0)
        val g = Mat()
        Core.subtract(gain, Scalar(1.0), g)
        Core.multiply(g, ringF, g)
        Core.add(g, Scalar(1.0), g)

        // 원본 해상도에서는 손가락 주변 영역만 처리
        val roiS = expand(Imgproc.boundingRect(ring8), 10, ss)
        val roiF = toFull(roiS, s, w, h)
        val out = img.clone()
        val imgRoi = Mat()
        img.submat(roiF).convertTo(imgRoi, CvType.CV_32FC3)
        val refUp = Mat()
        Imgproc.resize(ref.submat(roiS), refUp, roiF.size(), 0.0, 0.0, Imgproc.INTER_CUBIC)
        val gUp = Mat()
        Imgproc.resize(g.submat(roiS), gUp, roiF.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val soft = Mat()
        Imgproc.GaussianBlur(mask.submat(roiF), soft, Size(0.0, 0.0), 4.0)
        soft.convertTo(soft, CvType.CV_32F, 1.0 / 255.0)

        // out = (img*(1-soft) + ref*soft) * g
        val diff = Mat()
        Core.subtract(refUp, imgRoi, diff)
        Core.multiply(diff, merge3(soft), diff)
        Core.add(imgRoi, diff, imgRoi)
        Core.multiply(imgRoi, merge3(gUp), imgRoi)
        val roi8 = Mat()
        imgRoi.convertTo(roi8, CvType.CV_8UC3)
        roi8.copyTo(out.submat(roiF))

        listOf(small, m, paper, dil, valid8, valid, paperF, num, den, ref, paperS, paperSF, lumRef, lumP, gain,
            ring8, ringF, g, imgRoi, refUp, gUp, soft, diff, roi8).forEach { it.release() }
        return out
    }

    // ================================================================= 4. 화질 보정

    /** 종이 배경(조명·그림자)을 추정해 나눠서 배경을 균일한 흰색으로 */
    private fun flattenIllumination(img: Mat): Mat {
        val h = img.rows()
        val w = img.cols()
        val s = 1000.0 / max(h, w)
        val small = Mat()
        Imgproc.resize(img, small, Size((w * s).roundToInt().toDouble(), (h * s).roundToInt().toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val k = max(15, (max(small.rows(), small.cols()) * 0.035).toInt()) or 1
        val bg = Mat()
        Imgproc.morphologyEx(small, bg, Imgproc.MORPH_CLOSE, ellipse(k)) // 글자(어두운 것) 제거
        small.release()
        Imgproc.medianBlur(bg, bg, 21)
        Imgproc.GaussianBlur(bg, bg, Size(0.0, 0.0), 5.0)
        val bgFull = Mat()
        Imgproc.resize(bg, bgFull, img.size(), 0.0, 0.0, Imgproc.INTER_CUBIC)
        bg.release()
        Core.max(bgFull, Scalar(1.0, 1.0, 1.0), bgFull)
        val out = Mat()
        Core.divide(img, bgFull, out, 255.0) // 8비트 포화 연산: img / bg * 255
        bgFull.release()
        return out
    }

    private fun finish(norm: Mat, mode: ColorMode): Mat {
        if (mode == ColorMode.BW) {
            val g = Mat()
            Imgproc.cvtColor(norm, g, Imgproc.COLOR_RGB2GRAY)
            Imgproc.GaussianBlur(g, g, Size(0.0, 0.0), 0.8)
            val bw = Mat()
            Imgproc.adaptiveThreshold(g, bw, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 31, 12.0)
            // 뒷면 비침(종이보다 조금만 어두운 흐린 글씨)은 검게 찍지 않음
            val ghost = Mat()
            Imgproc.threshold(g, ghost, BW_GHOST - 1.0, 255.0, Imgproc.THRESH_BINARY)
            bw.setTo(Scalar(255.0), ghost)
            ghost.release()
            g.release()
            return bw
        }
        val x = if (mode == ColorMode.COLOR) norm.clone() else Mat().also { Imgproc.cvtColor(norm, it, Imgproc.COLOR_RGB2GRAY) }
        // 레벨: 종이는 완전 흰색, 글자는 더 진하게
        val small = Mat()
        val s = min(1.0, 1000.0 / max(x.rows(), x.cols()))
        Imgproc.resize(x, small, Size(), s, s, Imgproc.INTER_AREA)
        val sb = ByteArray((small.total() * small.channels()).toInt())
        small.get(0, 0, sb)
        small.release()
        val lo = percentileByHistogram(sb, 1.0)
        val hi = WHITE_POINT // 흰색 기준을 낮춰 뒷면 비침(밝기 200~235)도 지움
        val lutBytes = ByteArray(256) { v ->
            val t = ((v - lo) / max(hi - lo, 1.0)).coerceIn(0.0, 1.0)
            (t.pow(TONE_GAMMA) * 255.0).roundToInt().coerceIn(0, 255).toByte() // 감마>1: 옅은 글씨를 진하게
        }
        val lut = Mat(1, 256, CvType.CV_8UC1)
        lut.put(0, 0, lutBytes)
        val y = Mat()
        Core.LUT(x, lut, y)
        x.release(); lut.release()
        // 언샤프 마스크
        val blur = Mat()
        Imgproc.GaussianBlur(y, blur, Size(0.0, 0.0), 1.2)
        val out = Mat()
        Core.addWeighted(y, 1.6, blur, -0.6, 0.0, out)
        y.release(); blur.release()
        return out
    }

    /**
     * 가장자리 띠(각 변의 3%) 안의 긴 세로선·가로선(페이지 경계선, 접힌 선 그림자, 책 옆면 줄무늬 찌꺼기)을 흰색으로.
     * 글자는 짧은 획들이라 긴 선 모양 걸러내기(열림 연산)를 통과하지 못한다. out 을 직접 수정.
     */
    private fun cleanEdges(out: Mat) {
        val h = out.rows()
        val w = out.cols()
        val gray = Mat()
        if (out.channels() == 3) Imgproc.cvtColor(out, gray, Imgproc.COLOR_RGB2GRAY) else out.copyTo(gray)
        val s = min(1.0, 1200.0 / max(h, w))
        val g = Mat()
        if (s < 1.0) Imgproc.resize(gray, g, Size(rint(w * s).toDouble(), rint(h * s).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA) else gray.copyTo(g)
        gray.release()
        val sh = g.rows()
        val sw = g.cols()
        val dark = Mat()
        Imgproc.threshold(g, dark, 199.0, 255.0, Imgproc.THRESH_BINARY_INV) // g < 200
        g.release()
        val kv = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, max(5, (sh * 0.04).toInt()).toDouble()))
        val kh = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(max(5, (sw * 0.04).toInt()).toDouble(), 1.0))
        val vert = Mat()
        val horz = Mat()
        Imgproc.morphologyEx(dark, vert, Imgproc.MORPH_OPEN, kv)
        Imgproc.morphologyEx(dark, horz, Imgproc.MORPH_OPEN, kh)
        dark.release(); kv.release(); kh.release()
        val vb = ByteArray(sh * sw).also { vert.get(0, 0, it) }
        val hb = ByteArray(sh * sw).also { horz.get(0, 0, it) }
        vert.release(); horz.release()
        val bx = max(2, (sw * 0.03).toInt())
        val by = max(2, (sh * 0.03).toInt())
        val mb = ByteArray(sh * sw)
        var any = false
        for (y in 0 until sh) for (x in 0 until sw) {
            val k = y * sw + x
            val inX = x < bx || x >= sw - bx
            val inY = y < by || y >= sh - by
            if ((inX && vb[k].toInt() != 0) || (inY && hb[k].toInt() != 0)) {
                mb[k] = 255.toByte(); any = true
            }
        }
        if (!any) return
        val m = Mat(sh, sw, CvType.CV_8UC1).also { it.put(0, 0, mb) }
        val k5 = Mat.ones(5, 5, CvType.CV_8UC1)
        Imgproc.dilate(m, m, k5)
        k5.release()
        val full = Mat()
        if (s < 1.0) Imgproc.resize(m, full, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR) else m.copyTo(full)
        m.release()
        out.setTo(Scalar.all(255.0), full)
        full.release()
    }

    /** 손가락 영역을 흰색으로 채우고(경계는 부드럽게), 주변의 옅은 그림자·테두리도 정리. out 을 직접 수정. */
    private fun erase(out: Mat, mask: Mat) {
        val h = mask.rows()
        val w = mask.cols()
        val s = 600.0 / max(h, w)
        val ss = Size((w * s).roundToInt().toDouble(), (h * s).roundToInt().toDouble())
        val mS = Mat()
        Imgproc.resize(mask, mS, ss, 0.0, 0.0, Imgproc.INTER_NEAREST)
        val d = max(5, (min(ss.width, ss.height) * 0.04).toInt()) or 1
        val ringS = Mat()
        Imgproc.dilate(mS, ringS, ellipse(d))
        val roiS = expand(Imgproc.boundingRect(ringS), 6, ss)
        Imgproc.GaussianBlur(ringS, ringS, Size(0.0, 0.0), max(1.0, 6.0 * s))
        val roiF = toFull(roiS, s, w, h)

        val ring = Mat()
        Imgproc.resize(ringS.submat(roiS), ring, roiF.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        ring.convertTo(ring, CvType.CV_32F, 1.0 / 255.0)
        val soft = Mat()
        Imgproc.GaussianBlur(mask.submat(roiF), soft, Size(0.0, 0.0), 3.0)
        soft.convertTo(soft, CvType.CV_32F, 1.0 / 255.0)

        val outRoi = out.submat(roiF)
        val lum = Mat()
        if (out.channels() == 3) Imgproc.cvtColor(outRoi, lum, Imgproc.COLOR_RGB2GRAY) else outRoi.copyTo(lum)
        lum.convertTo(lum, CvType.CV_32F)
        // light = clip((lum - 150) / 60, 0, 1) * ring
        val light = Mat()
        Core.subtract(lum, Scalar(150.0), light)
        Core.multiply(light, Scalar(1.0 / 60.0), light)
        Core.max(light, Scalar(0.0), light)
        Core.min(light, Scalar(1.0), light)
        Core.multiply(light, ring, light)
        val wgt = Mat()
        Core.max(soft, light, wgt)

        // out = out + (255 - out) * wgt
        val f = Mat()
        outRoi.convertTo(f, if (out.channels() == 3) CvType.CV_32FC3 else CvType.CV_32F)
        val inv = Mat()
        Core.subtract(Mat(f.size(), f.type(), Scalar.all(255.0)), f, inv)
        Core.multiply(inv, if (out.channels() == 3) merge3(wgt) else wgt, inv)
        Core.add(f, inv, f)
        val back = Mat()
        f.convertTo(back, out.type())
        back.copyTo(outRoi)
        listOf(mS, ringS, ring, soft, lum, light, wgt, f, inv, back).forEach { it.release() }
    }

    // ================================================================= 유틸

    private fun ellipse(k: Int): Mat = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))

    private fun merge3(m: Mat): Mat {
        val out = Mat()
        Core.merge(listOf(m, m, m), out)
        return out
    }

    /** 3채널 float 의 채널 평균 → 1채널 float */
    private fun channelMean(m: Mat): Mat {
        val t = Mat(1, 3, CvType.CV_32F)
        t.put(0, 0, floatArrayOf(1f / 3, 1f / 3, 1f / 3))
        val out = Mat()
        Core.transform(m, out, t)
        t.release()
        return out
    }

    private fun expand(r: Rect, pad: Int, size: Size): Rect {
        val x0 = max(0, r.x - pad)
        val y0 = max(0, r.y - pad)
        val x1 = min(size.width.toInt(), r.x + r.width + pad)
        val y1 = min(size.height.toInt(), r.y + r.height + pad)
        return Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    }

    private fun toFull(r: Rect, s: Double, w: Int, h: Int): Rect {
        val x0 = floor(r.x / s).toInt().coerceIn(0, w - 1)
        val y0 = floor(r.y / s).toInt().coerceIn(0, h - 1)
        val x1 = ceil((r.x + r.width) / s).toInt().coerceIn(x0 + 1, w)
        val y1 = ceil((r.y + r.height) / s).toInt().coerceIn(y0 + 1, h)
        return Rect(x0, y0, x1 - x0, y1 - y0)
    }

    companion object {
        /** 반올림을 파이썬 round() 와 같게(0.5 는 짝수 쪽으로) — 기준 구현과 픽셀 단위로 맞추기 위함 */
        fun rint(x: Double): Int = Math.rint(x).toInt()

        const val MODEL_W = 488
        const val MODEL_H = 712
        const val GRID_W = 31
        const val GRID_H = 45

        /** 처리 해상도 상한(긴 변). 카메라 원본(4000x3000)을 그대로 써서 양면을 나눠도 글씨가 또렷하게 */
        const val MAX_SIDE = 4000

        /** 이보다 밝으면 흰 종이로(뒷면 비침 제거) */
        const val WHITE_POINT = 220.0

        /** 1보다 크면 중간 밝기(옅은 글씨)를 진하게 */
        const val TONE_GAMMA = 2.0

        /** 흑백 모드에서 이보다 밝은 것은 글자로 보지 않음 */
        const val BW_GHOST = 185.0

        /** 글줄 분석 해상도(긴 변) */
        const val ANALYSIS_SIDE = 1600.0

        /** 모델 격자를 위아래로 연장할 칸 수 */
        const val GRID_EXT = 1

        fun median(v: DoubleArray): Double = percentile(v, 50.0)

        fun percentile(v: DoubleArray, p: Double): Double {
            if (v.isEmpty()) return 0.0
            val c = v.copyOf()
            c.sort()
            val idx = (c.size - 1) * p / 100.0
            val lo = floor(idx).toInt()
            val hi = ceil(idx).toInt()
            return c[lo] + (c[hi] - c[lo]) * (idx - lo)
        }

        fun percentile(v: FloatArray, p: Double): Float {
            if (v.isEmpty()) return 0f
            val c = v.copyOf()
            c.sort()
            val idx = (c.size - 1) * p / 100.0
            val lo = floor(idx).toInt()
            val hi = ceil(idx).toInt()
            return (c[lo] + (c[hi] - c[lo]) * (idx - lo)).toFloat()
        }

        /** 8비트 값 배열의 백분위수 (히스토그램 방식) */
        fun percentileByHistogram(bytes: ByteArray, p: Double): Double {
            if (bytes.isEmpty()) return 0.0
            val hist = IntArray(256)
            for (b in bytes) hist[b.toInt() and 0xFF]++
            val target = (bytes.size - 1) * p / 100.0
            var acc = 0L
            for (v in 0 until 256) {
                acc += hist[v]
                if (acc > target) return v.toDouble()
            }
            return 255.0
        }

        fun gaussian1d(v: DoubleArray, sigma: Double): DoubleArray {
            val r = (sigma * 3).roundToInt()
            val k = DoubleArray(2 * r + 1) { kotlin.math.exp(-((it - r) * (it - r)) / (2 * sigma * sigma)) }
            val ks = k.sum()
            return DoubleArray(v.size) { i ->
                var sum = 0.0
                for (j in -r..r) {
                    val idx = (i + j).coerceIn(0, v.size - 1)
                    sum += v[idx] * k[j + r]
                }
                sum / ks
            }
        }

        /** 최소제곱 직선 x = a*y + b */
        fun polyfit1(ys: List<Double>, xs: List<Double>): Pair<Double, Double>? {
            val n = ys.size
            if (n < 2) return null
            val my = ys.average()
            val mx = xs.average()
            var sxy = 0.0
            var syy = 0.0
            for (i in 0 until n) {
                sxy += (ys[i] - my) * (xs[i] - mx)
                syy += (ys[i] - my) * (ys[i] - my)
            }
            if (syy == 0.0) return null
            val a = sxy / syy
            return Pair(a, mx - a * my)
        }
    }
}
