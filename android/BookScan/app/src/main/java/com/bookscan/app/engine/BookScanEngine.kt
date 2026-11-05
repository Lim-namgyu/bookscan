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
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
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
class BookScanEngine(modelBytes: ByteArray, regionModelBytes: ByteArray? = null) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        modelBytes,
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(4)
        }
    )

    /** 책 영역 찾기 (모델이 없으면 사진 전체를 책으로 봄) */
    private val region: BookRegion? = regionModelBytes?.let { runCatching { BookRegion(it) }.getOrNull() }

    override fun close() {
        session.close()
        region?.close()
    }

    // ================================================================= 공개 API

    /** 1단계: 사진(RGB) → 곡면 편 페이지 목록 (양면이면 2장) */
    fun dewarpPages(rgb: Mat, layout: Layout): List<DewarpedPage> {
        // 0) 책 영역: 책만 잘라내고(멀리서 찍어도 책이 화면을 채우게), 마스크는 바닥 지우기·접힌 선 찾기에 사용
        var src = rgb
        var bmask: Mat? = null
        val reg = region?.let { r -> runCatching { r.mask(rgb) }.getOrNull() }
        if (reg != null) {
            val (m, box) = reg
            val cb = BookRegion.cropBox(box, rgb.cols(), rgb.rows())
            if (cb != null) {
                src = rgb.cropClone(cb)
                bmask = m.cropClone(cb)
                m.release()
            } else {
                bmask = m
            }
        }
        try {
            // 폰을 세로로 든 채 펼친 책을 찍으면 글이 옆으로 누워 있음 → 글이 바로 서도록 먼저 돌림
            val rot = PageAnalysis.detectRotation(src)
            if (rot != 0) {
                val r = PageAnalysis.rotate(src, rot)
                if (src !== rgb) src.release()
                src = r
                bmask?.let { old ->
                    bmask = PageAnalysis.rotate(old, rot)
                    old.release()
                }
            }
            val gutter = if (layout == Layout.SINGLE) null else PageAnalysis.findGutter(src, layout == Layout.DOUBLE, bmask)
            val parts = if (gutter != null) splitSpread(src, gutter, bmask) else listOf(Pair(src.clone(), bmask?.clone()))
            return parts.mapIndexed { i, (p0, pm0) ->
                var p = p0
                val pm = BookRegion.hullMask(pm0)
                pm0?.release()
                var pmUse = pm
                if (pm != null) {
                    val iso = isolatePage(p, pm)
                    if (iso != null) {
                        p.release(); pm.release()
                        p = iso.first; pmUse = iso.second
                    }
                }
                // 곡면 펴기 모델은 세로로 긴 페이지를 기대한다. 가로로 인쇄된 쪽(표 등)을 돌려 세운 경우처럼
                // 페이지가 가로로 길면 세로로 눕혀서 편 뒤 다시 돌린다.
                val land = p.cols() > p.rows() * LANDSCAPE_RATIO
                if (land) {
                    val r = PageAnalysis.rotate(p, 90)
                    p.release(); p = r
                    pmUse?.let { old -> pmUse = PageAnalysis.rotate(old, 90); old.release() }
                }
                val flat0 = try {
                    rectify(p, pmUse)
                } finally {
                    p.release(); pmUse?.release()
                }
                val flat = if (land) PageAnalysis.rotate(flat0, 270).also { flat0.release() } else flat0
                val skip = if (parts.size == 1) null else if (i == 0) Side.RIGHT else Side.LEFT
                DewarpedPage(flat, skip)
            }
        } finally {
            if (src !== rgb) src.release()
            bmask?.release()
        }
    }

    /** 2단계: 손가락 제거 + 화질 보정. 결과는 RGB 8UC3(컬러) 또는 8UC1(회색·흑백) */
    fun clean(flat: Mat, mode: ColorMode, removeFingers: Boolean, skipSide: Side?): Mat {
        val mask = if (removeFingers) fingerMask(flat, skipSide) else null
        val hasMask = mask != null && Core.countNonZero(mask) > 0
        val filled = if (hasMask) fillPaper(flat, mask!!) else flat
        val (norm, bgSmall) = flattenIllumination(filled)
        if (filled !== flat) filled.release()
        val out = finish(norm, mode, bgSmall)
        norm.release(); bgSmall.release()
        if (hasMask) erase(out, mask!!)
        mask?.release()
        cleanEdges(out, skipSide) // 손가락 검사에서 뺀 쪽 = 접힌 선 쪽
        return out
    }

    // ================================================================= 1. 양면 분리

    /** 접힌 선 기준 좌/우 페이지(새 Mat). 반대쪽은 배경색으로 가린다. 책 마스크도 같은 방식으로 나눈다. */
    private fun splitSpread(img: Mat, gutter: Pair<Double, Double>, mask: Mat?): List<Pair<Mat, Mat?>> {
        val w = img.cols()
        val h = img.rows()
        val (a, b) = gutter
        val xTop = b
        val xBot = a * (h - 1) + b
        val bg = borderMedianColor(img)
        val result = ArrayList<Pair<Mat, Mat?>>()
        for (left in listOf(true, false)) {
            val poly = if (left) {
                MatOfPoint(Point(xTop, 0.0), Point(w.toDouble(), 0.0), Point(w.toDouble(), h.toDouble()), Point(xBot, h.toDouble()))
            } else {
                MatOfPoint(Point(0.0, 0.0), Point(xTop, 0.0), Point(xBot, h.toDouble()), Point(0.0, h.toDouble()))
            }
            val lo = if (left) 0 else max(0, min(xTop, xBot).toInt() - 10)
            val hi = if (left) min(w, max(xTop, xBot).toInt() + 10) else w
            val out = img.clone()
            Imgproc.fillPoly(out, listOf(poly), bg)
            val crop = out.cropClone(0, h, lo, hi)
            out.release()
            var pm: Mat? = null
            if (mask != null) {
                val mm = mask.clone()
                Imgproc.fillPoly(mm, listOf(poly), Scalar(0.0))
                pm = mm.cropClone(0, h, lo, hi)
                mm.release()
            }
            poly.release()
            result.add(Pair(crop, pm))
        }
        return result
    }

    /**
     * 곡면 펴기 모델 입력 정리: 페이지 상자(+여유)로 자르고, 페이지(조금 넓힌 마스크) 바깥은 단색으로 칠한다.
     * 바닥 무늬·옆 페이지가 사라져 모델이 페이지 경계를 정확히 잡는다. 반환: (새 페이지, 새 마스크) 또는 null(그대로 사용)
     */
    private fun isolatePage(p: Mat, pm: Mat): Pair<Mat, Mat>? {
        val h = p.rows()
        val w = p.cols()
        val r = Imgproc.boundingRect(pm)
        if (r.width.toLong() * r.height < 0.1 * w * h) return null
        val mx = (r.width * ISOLATE_MARGIN).toInt()
        val my = (r.height * ISOLATE_MARGIN).toInt()
        val x0 = max(0, r.x - mx)
        val y0 = max(0, r.y - my)
        val x1 = min(w, r.x + r.width + mx)
        val y1 = min(h, r.y + r.height + my)
        val roi = Rect(x0, y0, x1 - x0, y1 - y0)
        val p2 = p.cropClone(roi)
        val pm2 = pm.cropClone(roi)
        val d = max(3, (max(r.width, r.height) * ISOLATE_DILATE).toInt()) or 1
        val keep = Mat()
        Imgproc.dilate(pm2, keep, ellipse(d))
        val outside = Mat()
        Core.compare(keep, Scalar(0.0), outside, Core.CMP_EQ)
        p2.setTo(Scalar(ISOLATE_FILL, ISOLATE_FILL, ISOLATE_FILL), outside)
        keep.release(); outside.release()
        return Pair(p2, pm2)
    }

    /**
     * 곡면 편 페이지에서 책 바깥(바닥·책상 무늬)을 지운다 (새 Mat 또는 null=그대로).
     *  1) 마스크가 절반도 안 덮는 가장자리 행·열은 잘라내고
     *  2) 남은 부분 중 마스크 바깥은 주변 종이색으로 메운다(손가락 메우기와 같은 방법) → 보정 후 흰색.
     * 마스크 경계가 거칠어(320 격자) 조금 안쪽으로 줄여서 쓴다. 페이지 여백은 비어 있어 손실 없음.
     */
    private fun applyPageMask(flat: Mat, pm: Mat, erode: Boolean = true): Mat? {
        val h = pm.rows()
        val w = pm.cols()
        val inside = Mat()
        Imgproc.threshold(pm, inside, 127.0, 255.0, Imgproc.THRESH_BINARY)
        try {
            if (Core.countNonZero(inside) < 0.3 * w * h) return null // 마스크가 이상하면 손대지 않음
            if (erode) {
                val e = max(3, (min(h, w) * PAGE_MASK_ERODE).toInt()) or 1
                Imgproc.erode(inside, inside, ellipse(e))
            }
            val ib = ByteArray(w * h).also { inside.get(0, 0, it) }
            val colc = IntArray(w)
            val rowc = IntArray(h)
            for (y in 0 until h) for (x in 0 until w) if (ib[y * w + x].toInt() != 0) {
                colc[x]++; rowc[y]++
            }
            var x0 = -1; var x1 = -1; var nc = 0
            for (x in 0 until w) if (colc[x] >= 0.5 * h) {
                if (x0 < 0) x0 = x
                x1 = x + 1; nc++
            }
            var y0 = -1; var y1 = -1; var nr = 0
            for (y in 0 until h) if (rowc[y] >= 0.5 * w) {
                if (y0 < 0) y0 = y
                y1 = y + 1; nr++
            }
            if (nc < w * 0.5 || nr < h * 0.5) return null
            val roi = Rect(x0, y0, x1 - x0, y1 - y0)
            val res = flat.cropClone(roi)
            val outside = Mat()
            inside.withRoi(roi) { Core.compare(it, Scalar(0.0), outside, Core.CMP_EQ) }
            return if (Core.countNonZero(outside) > 0) {
                val filled = fillPaper(res, outside)
                res.release()
                filled
            } else {
                res
            }.also { outside.release() }
        } finally {
            inside.release()
        }
    }

    private fun borderMedianColor(img: Mat): Scalar {
        val w = img.cols()
        val rows = min(8, img.rows() / 2)
        val n = rows * w * 3
        val top = ByteArray(n)
        val bot = ByteArray(n)
        img.cropClone(0, rows, 0, w).also { it.get(0, 0, top); it.release() }
        img.cropClone(img.rows() - rows, img.rows(), 0, w).also { it.get(0, 0, bot); it.release() }
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
    private fun rectify(img: Mat, mask: Mat?): Mat {
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
        var pxE = extendGridV(gx, GRID_EXT)
        var pyE = extendGridV(gy, GRID_EXT)
        px.release(); py.release()
        val ghE = GRID_H + 2 * GRID_EXT
        // 모델이 잡은 좌우 경계가 글자를 가로지르면(페이지가 사진 밖으로 잘렸거나 경계를 안쪽에 잡음) 그쪽으로 연장
        val (extL, extR) = extendGridToText(pxE, pyE, ghE, GRID_W, img)
        var gwE = GRID_W
        if (extL + extR > 0) {
            pxE = extendGridH(pxE, ghE, GRID_W, extL, extR)
            pyE = extendGridH(pyE, ghE, GRID_W, extL, extR)
            gwE = GRID_W + extL + extR
        }
        px = Mat(ghE, gwE, CvType.CV_32F).also { it.put(0, 0, pxE) }
        py = Mat(ghE, gwE, CvType.CV_32F).also { it.put(0, 0, pyE) }
        oh = oh * (ghE - 1.0) / (GRID_H - 1.0)
        val owE = ow * (gwE - 1.0) / (GRID_W - 1.0)
        val outW = max(16, rint(owE * scale))
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
        val fa = flatA.cropClone(t, ah - bt, l, aw - r)
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

        // ---- 가로세로 비율 보정: 모델의 격자 길이로 잰 페이지 폭은 휘어 기울어진 부분이 사진에서 짧게 찍혀
        // 실제보다 좁게 나온다(글자가 세로로 길쭉해짐). 지도의 국소 배율(가로/세로)이 평평한 곳에서 1이 되도록 폭을 맞춘다.
        val rAs = aspectRatioFix(fxa, fya)
        // ---- 원본 해상도로 확대 후 한 번만 재샘플링
        val fullW = max(16, rint(fw / a * rAs))
        val fullH = max(16, rint(fh / a))
        val fx = Mat()
        val fy = Mat()
        Imgproc.resize(fxa, fx, Size(fullW.toDouble(), fullH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        Imgproc.resize(fya, fy, Size(fullW.toDouble(), fullH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        fxa.release(); fya.release()
        val out = Mat()
        Imgproc.remap(img, out, fx, fy, Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE, Scalar(0.0))
        var pmOut: Mat? = null
        if (mask != null) {
            pmOut = Mat()
            Imgproc.remap(mask, pmOut, fx, fy, Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0))
        }
        // 사진 밖(격자를 사진 끝 너머로 연장한 곳)은 가장자리 픽셀이 늘어나 줄무늬가 되므로 페이지 밖으로 본다
        val inX = Mat()
        val inY = Mat()
        Core.inRange(fx, Scalar(0.0), Scalar(w - 1.0), inX)
        Core.inRange(fy, Scalar(0.0), Scalar(h - 1.0), inY)
        fx.release(); fy.release()
        Core.bitwise_and(inX, inY, inX)
        inY.release()
        val k5 = Mat.ones(5, 5, CvType.CV_8UC1)
        Imgproc.erode(inX, inX, k5)
        k5.release()
        val hasOutside = Core.countNonZero(inX) < inX.total()
        if (hasOutside) {
            if (pmOut == null) pmOut = inX else {
                Core.min(pmOut, inX, pmOut); inX.release()
            }
        } else {
            inX.release()
        }
        val (t2, b2, l2, r2) = trimRect(out)
        val roi = Rect(l2, t2, out.cols() - l2 - r2, out.rows() - t2 - b2)
        val trimmed = if (t2 + b2 + l2 + r2 == 0) out else out.cropClone(roi).also { out.release() }
        if (pmOut == null) return trimmed
        val pmT = if (t2 + b2 + l2 + r2 == 0) pmOut else pmOut.cropClone(roi).also { pmOut.release() }
        val masked = applyPageMask(trimmed, pmT, erode = mask != null)
        pmT.release()
        if (masked == null) return trimmed
        trimmed.release()
        return masked
    }

    /** bookscan._text_components: 글자 크기 덩어리의 상자 [x, y, w, h] (원본 좌표). 큰 덩어리(그림·상자)는 제외 */
    private fun textComponentBoxes(img: Mat): List<DoubleArray> {
        val h = img.rows()
        val w = img.cols()
        val s = 1000.0 / max(h, w)
        val small = Mat()
        if (s < 1.0) Imgproc.resize(img, small, Size(0.0, 0.0), s, s, Imgproc.INTER_AREA) else img.copyTo(small)
        val g = Mat()
        if (small.channels() == 3) Imgproc.cvtColor(small, g, Imgproc.COLOR_RGB2GRAY) else small.copyTo(g)
        small.release()
        val block = max(15, (min(g.rows(), g.cols()) * 0.03).toInt()) or 1
        val m = Mat()
        Imgproc.adaptiveThreshold(g, m, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, block, 12.0)
        val lbl = Mat()
        val st = Mat()
        val cen = Mat()
        val n = Imgproc.connectedComponentsWithStats(m, lbl, st, cen, 8, CvType.CV_32S)
        val stats = IntArray(n * 5).also { st.get(0, 0, it) }
        listOf(g, m, lbl, st, cen).forEach { it.release() }
        val gh = (h * s).roundToInt().coerceAtLeast(1)
        val gw = (w * s).roundToInt().coerceAtLeast(1)
        val out = ArrayList<DoubleArray>()
        val sc = if (s < 1.0) 1.0 / s else 1.0
        for (i in 1 until n) {
            val hh = stats[i * 5 + Imgproc.CC_STAT_HEIGHT]
            val ww = stats[i * 5 + Imgproc.CC_STAT_WIDTH]
            if (hh >= 3 && hh <= gh * 0.03 && ww <= gw * 0.06) {
                out.add(doubleArrayOf(stats[i * 5] * sc, stats[i * 5 + 1] * sc, ww * sc, hh * sc))
            }
        }
        return out
    }

    /**
     * bookscan.extend_grid_to_text: 격자의 왼쪽/오른쪽 경계선이 글자를 가로지르면(경계 양쪽에 걸친 글자 덩어리가
     * 여러 글줄에 걸쳐 있음) 그쪽으로 연장할 칸 수. 반환: (왼쪽, 오른쪽)
     */
    private fun extendGridToText(px: FloatArray, py: FloatArray, rows: Int, cols: Int, img: Mat): Pair<Int, Int> {
        val h = img.rows()
        val w = img.cols()
        var boxes: List<DoubleArray>? = null
        val ext = intArrayOf(0, 0)
        for (side in 0..1) {
            val c = if (side == 0) 0 else cols - 1
            val bx = DoubleArray(rows) { px[it * cols + c].toDouble() }
            val by = DoubleArray(rows) { py[it * cols + c].toDouble() }
            if (boxes == null) {
                boxes = textComponentBoxes(img)
                if (boxes.isEmpty()) break
            }
            val order = by.indices.sortedBy { by[it] }
            val bys = DoubleArray(rows) { by[order[it]] }
            val bxs = DoubleArray(rows) { bx[order[it]] }
            var nStr = 0
            var rMin = Double.MAX_VALUE
            var rMax = -Double.MAX_VALUE
            for (b in boxes) {
                val cy = b[1] + b[3] / 2.0
                val xb = Straightener.interpPublic(cy, bys, bxs)
                if (b[0] < xb - 1.0 && b[0] + b[2] > xb + 1.0 && b[2] >= 3) {
                    nStr++; rMin = min(rMin, cy); rMax = max(rMax, cy)
                }
            }
            if (nStr < EXT_STRADDLE_MIN || rMax - rMin < h * 0.15) continue
            val need: Double
            if (side == 0) {
                var step = 0.0
                for (r in 0 until rows) step += abs(px[r * cols] - px[r * cols + 1])
                step /= rows
                need = bx.max() / max(step, 1.0)
            } else {
                var step = 0.0
                for (r in 0 until rows) step += abs(px[r * cols + cols - 1] - px[r * cols + cols - 2])
                step /= rows
                need = (w - 1 - bx.min()) / max(step, 1.0)
            }
            val k = min(GRID_EXT_H_MAX.toDouble(), ceil(need)).toInt()
            if (k > 0) ext[side] = k
        }
        return Pair(ext[0], ext[1])
    }

    /** 격자를 좌우로 직선 연장 (가장자리 두 열의 차이만큼 계속) */
    private fun extendGridH(g: FloatArray, rows: Int, cols: Int, left: Int, right: Int): FloatArray {
        val nc = cols + left + right
        val out = FloatArray(rows * nc)
        for (r in 0 until rows) {
            for (c in 0 until cols) out[r * nc + left + c] = g[r * cols + c]
            for (j in 1..left) out[r * nc + left - j] = g[r * cols] + j * (g[r * cols] - g[r * cols + 1])
            for (j in 1..right) out[r * nc + left + cols - 1 + j] = g[r * cols + cols - 1] + j * (g[r * cols + cols - 1] - g[r * cols + cols - 2])
        }
        return out
    }

    /**
     * bookscan.aspect_ratio_fix: 출력 폭에 곱할 비율. mx, my: 출력(분석 해상도) 픽셀 → 원본 사진 좌표 지도(CV_32F).
     * 가로 한 칸이 사진에서 차지하는 길이 / 세로 한 칸이 차지하는 길이 를 페이지 안쪽에서 재고, 평평한 곳(비가 큰 쪽)의 값을 쓴다.
     */
    private fun aspectRatioFix(mx: Mat, my: Mat): Double {
        val h = mx.rows()
        val w = mx.cols()
        val st = max(4, min(h, w) / 40)
        val ys = ArrayList<Int>()
        val xs = ArrayList<Int>()
        var y = (h * 0.1).toInt()
        while (y < (h * 0.9).toInt() - st) { ys.add(y); y += st }
        var x = (w * 0.1).toInt()
        while (x < (w * 0.9).toInt() - st) { xs.add(x); x += st }
        if (ys.size < 3 || xs.size < 3) return 1.0
        val ax = FloatArray(w * h).also { mx.get(0, 0, it) }
        val ay = FloatArray(w * h).also { my.get(0, 0, it) }
        val ratio = DoubleArray(ys.size * xs.size)
        var k = 0
        for (yy in ys) for (xx in xs) {
            val i = yy * w + xx
            val ju = hypot((ax[i + st] - ax[i]).toDouble(), (ay[i + st] - ay[i]).toDouble())
            val jv = hypot((ax[i + st * w] - ax[i]).toDouble(), (ay[i + st * w] - ay[i]).toDouble())
            ratio[k++] = ju / max(jv, 1e-6)
        }
        return percentile(ratio, ASPECT_PCT).coerceIn(ASPECT_MIN, ASPECT_MAX)
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
        val full = Mat()
        if (img.channels() == 3) Imgproc.cvtColor(img, full, Imgproc.COLOR_RGB2GRAY) else img.copyTo(full)
        val gray = Mat()
        Imgproc.resize(full, gray, Size(rint(w * s).toDouble(), rint(h * s).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        full.release()
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
        // 결 방향: 책 옆면(종이 단면)은 변과 나란한 줄무늬라 한 방향 기울기만 강하다. 글자는 두 방향 다 있음
        val colGx = DoubleArray(sw)
        val colGy = DoubleArray(sw)
        for (y in r0 until r1) for (x in 0 until sw) {
            colGx[x] += abs(gx[y * sw + x]).toDouble(); colGy[x] += abs(gy[y * sw + x]).toDouble()
        }
        val rowGx = DoubleArray(sh)
        val rowGy = DoubleArray(sh)
        for (y in 0 until sh) for (x in c0 until c1) {
            rowGx[y] += abs(gx[y * sw + x]).toDouble(); rowGy[y] += abs(gy[y * sw + x]).toDouble()
        }
        /**
         * 바깥에서 안으로 훑다가 빈 여백이 시작되는 곳(i)까지가 잘라낼 후보 띠.
         *  - 띠 전체가 무늬(카펫 등): 한도 10%
         *  - 띠가 변과 나란한 줄무늬(책 옆면, 두꺼운 책은 폭이 넓음): 한도 15%
         * v: 빈 비율, along/across: 변과 수직/나란한 방향 기울기 합 (모두 바깥→안 순서로 꺼냄)
         */
        fun tex(v: (Int) -> Double, along: (Int) -> Double, across: (Int) -> Double, n: Int, limit: Int): Int {
            for (i in 0 until min((limit * 1.5).toInt(), n - 3)) {
                if (v(i) >= 0.75 && v(i + 1) >= 0.75 && v(i + 2) >= 0.75) {
                    if (i == 0) return 0
                    var sum = 0.0
                    var sa = 0.0
                    var sc = 0.0
                    for (j in 0 until i) {
                        sum += v(j); sa += along(j); sc += across(j)
                    }
                    if (i <= limit && sum / i < 0.3) return i // 잘라낼 띠가 '무늬'일 때만
                    if (sa > STREAK_RATIO * sc) return i      // 변과 나란한 줄무늬(책 옆면)
                    return 0
                }
            }
            return 0
        }
        val limH2 = (sh * 0.10).toInt()
        val limW2 = (sw * 0.10).toInt()
        t = max(t, tex({ rowb[it] }, { rowGy[it] }, { rowGx[it] }, sh, limH2))
        bt = max(bt, tex({ rowb[sh - 1 - it] }, { rowGy[sh - 1 - it] }, { rowGx[sh - 1 - it] }, sh, limH2))
        l = max(l, tex({ colb[it] }, { colGx[it] }, { colGy[it] }, sw, limW2))
        r = max(r, tex({ colb[sw - 1 - it] }, { colGx[sw - 1 - it] }, { colGy[sw - 1 - it] }, sw, limW2))

        // 3) 페이지 가장자리 띠: 페이지가 들려 밑에 깔린 종이들의 단면·인쇄가 보이거나 옆 페이지 테두리가 들어온 것.
        //    바깥에서 안으로 훑어 '종이가 아닌 열'(어둡거나 색이 있음)이 이어지다가 흰 여백이 시작되는 곳까지 자른다.
        //    띠 안에 글자가 있으면(색 상자 안 글 등) 자르지 않는다.
        val paper = percentileByHistogram(g, 70.0)
        val chroma = FloatArray(sh * sw)
        val txt = ByteArray(sh * sw)
        if (img.channels() == 3) {
            val smallC = Mat()
            Imgproc.resize(img, smallC, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            val lab = Mat()
            Imgproc.cvtColor(smallC, lab, Imgproc.COLOR_RGB2Lab)
            // 종이 색은 조명 때문에 자리마다 조금씩 다르다 → 종이 픽셀의 a/b 를 넓게 평균한 '그 자리 종이 색' 기준 채도
            val ch = ArrayList<Mat>()
            Core.split(lab, ch)
            lab.release()
            val pp = Mat(sh, sw, CvType.CV_32F).also { m -> m.put(0, 0, FloatArray(sh * sw) { if (all[it] >= paper * 0.9) 1f else 0f }) }
            val sig = max(sw, sh) * 0.12
            val den = Mat()
            Imgproc.GaussianBlur(pp, den, Size(0.0, 0.0), sig)
            Core.add(den, Scalar(1e-3), den)
            val aF = Mat(); val bF = Mat()
            ch[1].convertTo(aF, CvType.CV_32F); ch[2].convertTo(bF, CvType.CV_32F)
            ch.forEach { it.release() }
            val aL = Mat(); val bL = Mat()
            Core.multiply(aF, pp, aL); Core.multiply(bF, pp, bL)
            Imgproc.GaussianBlur(aL, aL, Size(0.0, 0.0), sig); Imgproc.GaussianBlur(bL, bL, Size(0.0, 0.0), sig)
            Core.divide(aL, den, aL); Core.divide(bL, den, bL)
            Core.subtract(aF, aL, aF); Core.subtract(bF, bL, bF)
            Core.magnitude(aF, bF, aF)
            aF.get(0, 0, chroma)
            listOf(pp, den, aF, bF, aL, bL).forEach { it.release() }
            // 글자 마스크(글자 크기 덩어리)
            val gS = Mat()
            Imgproc.cvtColor(smallC, gS, Imgproc.COLOR_RGB2GRAY)
            smallC.release()
            val block = max(15, (min(sh, sw) * 0.03).toInt()) or 1
            val m = Mat()
            Imgproc.adaptiveThreshold(gS, m, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, block, 12.0)
            gS.release()
            val lbl = Mat(); val st = Mat(); val cen = Mat()
            val n = Imgproc.connectedComponentsWithStats(m, lbl, st, cen, 8, CvType.CV_32S)
            m.release(); cen.release()
            val stats = IntArray(n * 5).also { st.get(0, 0, it) }
            st.release()
            val labels = IntArray(sh * sw).also { lbl.get(0, 0, it) }
            lbl.release()
            val keep = BooleanArray(n) { i ->
                i != 0 && stats[i * 5 + 3] >= 3 && stats[i * 5 + 3] <= sh * 0.03 && stats[i * 5 + 2] <= sw * 0.06
            }
            for (i in 0 until sh * sw) if (keep[labels[i]]) txt[i] = 1
        }
        val medC = DoubleArray(sw) { x -> median(DoubleArray(r1 - r0) { all[(r0 + it) * sw + x] }) / paper }
        val chrC = DoubleArray(sw) { x -> median(DoubleArray(r1 - r0) { chroma[(r0 + it) * sw + x].toDouble() }) }
        val txtC = DoubleArray(sw) { x -> var c = 0; for (y in r0 until r1) c += txt[y * sw + x]; c.toDouble() / max(1, r1 - r0) }
        val medR = DoubleArray(sh) { y -> median(DoubleArray(c1 - c0) { all[y * sw + c0 + it] }) / paper }
        val chrR = DoubleArray(sh) { y -> median(DoubleArray(c1 - c0) { chroma[y * sw + c0 + it].toDouble() }) }
        val txtR = DoubleArray(sh) { y -> var c = 0; for (x in c0 until c1) c += txt[y * sw + x]; c.toDouble() / max(1, c1 - c0) }
        /** 바깥(0)에서 안으로: 흰 여백이 run 만큼 이어지는 첫 자리까지가 띠 후보. 띠 안에 확실히 종이가 아닌 열이
         *  run 폭 이상 이어져야 하고 글자가 없어야 한다. 확실한 띠가 끝나는 곳(+여유)까지만 자른다 */
        fun band(med: (Int) -> Double, chr: (Int) -> Double, tx: (Int) -> Double, n: Int, limit: Int, run: Int): Int {
            fun page(i: Int) = med(i) >= EDGE_PAPER_MIN && chr(i) <= EDGE_PAPER_CHROMA
            var i = 0
            while (i < min(limit, n - run)) {
                var all = true
                for (j in i until i + run) if (!page(j)) { all = false; break }
                if (all) break
                i++
            }
            if (i == 0 || i >= limit || i >= n - run) return 0
            val st = IntArray(i) { if (med(it) < EDGE_BAND_DARK || chr(it) > EDGE_BAND_CHROMA) 1 else 0 }
            var best = 0.0
            var last = -1
            for (c in 0 until i) {
                var cnt = 0
                for (j in c - run / 2 until c - run / 2 + run) if (j in 0 until i && st[j] == 1) cnt++
                best = max(best, cnt.toDouble() / run)
                if (st[c] == 1) last = c
            }
            if (best < 0.6) return 0
            var tsum = 0.0
            for (j in 0 until i) tsum += tx(j)
            if (tsum / i > EDGE_BAND_TEXT) return 0
            return last + 1 + run / 3
        }
        val runW = max(3, (sw * 0.015).toInt())
        val runH = max(3, (sh * 0.015).toInt())
        val limEW = (sw * EDGE_BAND_MAX).toInt()
        val limEH = (sh * EDGE_BAND_MAX).toInt()
        l = max(l, band({ medC[it] }, { chrC[it] }, { txtC[it] }, sw, limEW, runW))
        r = max(r, band({ medC[sw - 1 - it] }, { chrC[sw - 1 - it] }, { txtC[sw - 1 - it] }, sw, limEW, runW))
        t = max(t, band({ medR[it] }, { chrR[it] }, { txtR[it] }, sh, limEH, runH))
        bt = max(bt, band({ medR[sh - 1 - it] }, { chrR[sh - 1 - it] }, { txtR[sh - 1 - it] }, sh, limEH, runH))
        return Trim(rint(t / s), rint(bt / s), rint(l / s), rint(r / s))
    }

    // ================================================================= 3. 손가락 제거

    /** 종이보다 붉은(피부색) 두꺼운 덩어리 중 페이지 가장자리에서 들어온 것 → 손가락 마스크(원본 크기, 8UC1) */
    /**
     * bookscan._finger_bulge: 변에 닿은 피부색 덩어리(0/255, sh x sw)에서 안쪽으로 볼록 튀어나온 손가락 부분 또는 null.
     * side: 0=왼쪽 1=오른쪽 2=위 3=아래.
     * 변을 따라 행(또는 열)마다 '변에서 얼마나 깊이 들어왔나'를 재고, 가장 깊은 곳의 절반 이상 들어온 구간(손가락 몸통)이
     * 짧고 깊으면 손가락. 몸통 구간 앞뒤로 조금 넓혀(손가락 끝 둥근 부분) 그 부분만 돌려준다.
     */
    private fun fingerBulge(c: ByteArray, sh: Int, sw: Int, side: Int): ByteArray? {
        val vertical = side <= 1 // 왼쪽·오른쪽 변: 행을 따라감
        val nAlong = if (vertical) sh else sw
        val nDepth = if (vertical) sw else sh
        fun at(a: Int, d: Int): Boolean { // a: 변을 따라 위치, d: 변에서 깊이(0 = 변)
            val idx = when (side) {
                0 -> a * sw + d
                1 -> a * sw + sw - 1 - d
                2 -> d * sw + a
                else -> (sh - 1 - d) * sw + a
            }
            return c[idx].toInt() != 0
        }
        val depth = IntArray(nAlong)
        var dmax = 0
        for (a in 0 until nAlong) {
            for (d in nDepth - 1 downTo 0) if (at(a, d)) {
                depth[a] = d + 1; break
            }
            dmax = max(dmax, depth[a])
        }
        if (dmax == 0 || dmax < FINGER_MIN_DEPTH * nDepth) return null
        var best = 0
        var cur = 0
        var b0 = 0
        var c0 = 0
        for (a in 0 until nAlong) {
            if (depth[a] >= 0.5 * dmax) {
                if (cur == 0) c0 = a
                cur++
                if (cur > best) {
                    best = cur; b0 = c0
                }
            } else {
                cur = 0
            }
        }
        if (best > FINGER_MAX_ALONG * nAlong || best > FINGER_ALONG_DEPTH * dmax) return null
        val pad = (0.6 * dmax).toInt()
        val r0 = max(0, b0 - pad)
        val r1 = min(nAlong, b0 + best + pad)
        val out = ByteArray(sh * sw)
        for (y in 0 until sh) for (x in 0 until sw) {
            val a = if (vertical) y else x
            if (a in r0 until r1 && c[y * sw + x].toInt() != 0) out[y * sw + x] = 255.toByte()
        }
        return out
    }

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
        if (paperA.isEmpty()) return Mat.zeros(h, w, CvType.CV_8UC1) // 밝기가 고른 이미지: 종이 기준을 못 잡음
        val a0 = percentile(paperA.toFloatArray(), 50.0)

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
            // 심지를 원래 두께만큼 되살리되 피부색 영역 안에서만
            Core.compare(labels, Scalar(i.toDouble()), comp, Core.CMP_EQ)
            Imgproc.dilate(comp, comp, kern)
            Core.bitwise_and(comp, skin, comp)
            // 가장자리에 닿아 있어야 하고(손가락은 바깥에서 들어온다), 안쪽으로 볼록 튀어나온 부분만 손가락.
            // 손가락이 책 옆면 띠(변을 따라 길고 얕은 피부색 띠)와 붙어 있으면 띠 부분은 떼어낸다.
            var cb: ByteArray? = null
            val touches = listOf(
                Side.LEFT to (x <= edge), Side.RIGHT to (x + bw >= sw - edge),
                null to (y <= edge), null to (y + bh >= sh - edge)
            )
            for ((idx, t) in touches.withIndex()) {
                val (side, touch) = t
                if (!touch || (side != null && side == skipSide)) continue
                if (cb == null) cb = ByteArray(n).also { comp.get(0, 0, it) }
                val part = fingerBulge(cb, sh, sw, idx) ?: continue
                val pm = Mat(sh, sw, CvType.CV_8UC1).also { it.put(0, 0, part) }
                Core.bitwise_or(mask, pm, mask)
                pm.release()
            }
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

    /**
     * 마스크 자리를 주변 종이색으로 메우고, 바로 옆 그림자를 밝힌다(조명 보정 전에 해야 얼룩이 안 남음).
     * 손가락 자리와 책 바깥(곡면 편 결과의 가장자리)에 쓴다. 새 Mat 반환.
     * 큰 페이지에서도 메모리를 적게 쓰도록 원본 해상도 계산은 채널 하나씩 한다.
     */
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

        // 글자를 지운 '종이 배경' → 마스크 밖 종이만으로 가중 평균(정규화 컨볼루션)해 종이색 추정
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
        merge3(valid).also { v3 -> Core.multiply(paperF, v3, num); v3.release() }
        Imgproc.GaussianBlur(num, num, Size(0.0, 0.0), sig)
        val den = Mat()
        Imgproc.GaussianBlur(valid, den, Size(0.0, 0.0), sig)
        Core.max(den, Scalar(1e-3), den)
        val ref = Mat()
        merge3(den).also { d3 -> Core.divide(num, d3, ref); d3.release() }

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

        // 원본 해상도에서는 마스크 주변 영역만, 채널 하나씩: out = (img*(1-soft) + ref*soft) * g
        // 축소하며 사라진 가는 마스크 조각(가장자리 쐐기 끝 등)도 처리 범위에 들도록 원본 마스크 상자와 합침
        val roiS = expand(union(Imgproc.boundingRect(ring8), toSmall(Imgproc.boundingRect(mask), s)), 10, ss)
        val roiF = toFull(roiS, s, w, h)
        val out = img.clone()
        val gUp = Mat()
        g.withRoi(roiS) { Imgproc.resize(it, gUp, roiF.size(), 0.0, 0.0, Imgproc.INTER_LINEAR) }
        val soft = Mat()
        mask.withRoi(roiF) { Imgproc.GaussianBlur(it, soft, Size(0.0, 0.0), 4.0) }
        soft.convertTo(soft, CvType.CV_32F, 1.0 / 255.0)
        val refCh = ArrayList<Mat>()
        Core.split(ref, refCh)
        val imgC = Mat()
        val refUp = Mat()
        val tmp = Mat()
        out.withRoi(roiF) { outRoi ->
            for (c in 0 until 3) {
                Core.extractChannel(outRoi, imgC, c)
                imgC.convertTo(imgC, CvType.CV_32F)
                refCh[c].withRoi(roiS) { Imgproc.resize(it, refUp, roiF.size(), 0.0, 0.0, Imgproc.INTER_CUBIC) }
                Core.subtract(refUp, imgC, tmp)
                Core.multiply(tmp, soft, tmp)
                Core.add(imgC, tmp, imgC)
                Core.multiply(imgC, gUp, imgC)
                imgC.convertTo(tmp, CvType.CV_8U)
                Core.insertChannel(tmp, outRoi, c)
            }
        }
        refCh.forEach { it.release() }
        listOf(small, m, paper, dil, valid8, valid, paperF, num, den, ref, paperS, paperSF, lumRef, lumP, gain,
            ring8, ringF, g, gUp, soft, imgC, refUp, tmp).forEach { it.release() }
        return out
    }

    // ================================================================= 4. 화질 보정

    /** 종이 배경(조명·그림자)을 추정해 나눠서 배경을 균일한 흰색으로 */
    /** 반환: (평탄화한 이미지, 추정한 종이 밝기 지도 — 긴 변 1000px, 회색 CV_32F). 밝기 지도는 반사로 밝은 곳 판단에 쓴다 */
    private fun flattenIllumination(img: Mat): Pair<Mat, Mat> {
        val h = img.rows()
        val w = img.cols()
        val s = 1000.0 / max(h, w)
        val small = Mat()
        Imgproc.resize(img, small, Size((w * s).roundToInt().toDouble(), (h * s).roundToInt().toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        // 종이 결·잡티의 밝은 점이 배경을 실제보다 밝게 끌어올리지 않도록 살짝 흐린 뒤 배경을 잰다
        // (안 그러면 그림자 진 곳의 종이가 회색으로 남아 뒷면 비침과 구분이 안 됨)
        Imgproc.GaussianBlur(small, small, Size(0.0, 0.0), FLAT_PREBLUR)
        val k = max(15, (max(small.rows(), small.cols()) * 0.035).toInt()) or 1
        val bg = Mat()
        Imgproc.morphologyEx(small, bg, Imgproc.MORPH_CLOSE, ellipse(k)) // 글자(어두운 것) 제거
        small.release()
        Imgproc.medianBlur(bg, bg, 21)
        Imgproc.GaussianBlur(bg, bg, Size(0.0, 0.0), 5.0)
        val bgFull = Mat()
        Imgproc.resize(bg, bgFull, img.size(), 0.0, 0.0, Imgproc.INTER_CUBIC)
        Core.max(bgFull, Scalar(1.0, 1.0, 1.0), bgFull)
        val out = Mat()
        Core.divide(img, bgFull, out, 255.0) // 8비트 포화 연산: img / bg * 255
        bgFull.release()
        // 어두운 곳은 밝히면서 노이즈도 같이 커진다(배율 = 255/배경 밝기) → 배율이 큰 곳만 살짝 흐려 잡음을 누른다
        //   wn = clip((255/mean(bg) - GAIN0) / SPAN, 0, 1) ;  out += (blur(out) - out) * wn
        val bgGray = Mat()
        Imgproc.cvtColor(bg, bgGray, Imgproc.COLOR_RGB2GRAY)
        bg.release()
        val bgSmall = Mat()
        bgGray.convertTo(bgSmall, CvType.CV_32F)
        bgGray.release()
        val wnS = bgSmall.clone()
        Core.max(wnS, Scalar(1.0), wnS)
        Core.divide(255.0, wnS, wnS)
        wnS.convertTo(wnS, CvType.CV_32F, 1.0 / DENOISE_GAIN_SPAN, -DENOISE_GAIN0 / DENOISE_GAIN_SPAN)
        Core.max(wnS, Scalar(0.0), wnS)
        Core.min(wnS, Scalar(1.0), wnS)
        val mm = Core.minMaxLoc(wnS)
        if (mm.maxVal > 0.0) {
            val wn = Mat()
            Imgproc.resize(wnS, wn, img.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val blur = Mat()
            Imgproc.GaussianBlur(out, blur, Size(0.0, 0.0), DENOISE_SIGMA)
            val o = Mat()
            val b = Mat()
            val c8 = Mat()
            for (c in 0 until 3) {
                Core.extractChannel(out, c8, c)
                c8.convertTo(o, CvType.CV_32F)
                Core.extractChannel(blur, c8, c)
                c8.convertTo(b, CvType.CV_32F)
                Core.subtract(b, o, b)
                Core.multiply(b, wn, b)
                Core.add(o, b, o)
                o.convertTo(c8, CvType.CV_8U)
                Core.insertChannel(c8, out, c)
            }
            listOf(wn, blur, o, b, c8).forEach { it.release() }
        }
        wnS.release()
        return Pair(out, bgSmall)
    }

    // ================================================================= 글자 진하기 자리마다 맞춤

    private class LocalInk(val lv: FloatArray, val ref: Double, val smax: FloatArray)

    /**
     * 글자 진하기 지도(gs 크기). 초점이 안 맞거나 빛이 비쳐 글자가 옅게 찍힌 곳은 값이 높다.
     * 글자 크기 덩어리(그림·괘선·망점 제외, 색 글자 제외)의 픽셀만 모아 칸마다 하위 백분위 밝기를 재고 매끈하게 잇는다.
     * glare: 반사 정도(0~1, gs 크기) — 반사가 있는 칸은 더 옅은 글자도 글자로 보고 더 크게 키운다.
     * (engine/bookscan.py local_ink_level 와 같음)
     */
    private fun localInkLevel(gsM: Mat, gs: ByteArray, paper: Double, chromaS: FloatArray?, glare: FloatArray?): LocalInk? {
        val h = gsM.rows()
        val w = gsM.cols()
        val block = max(15, (min(h, w) * 0.03).toInt()) or 1
        val m = Mat()
        Imgproc.adaptiveThreshold(gsM, m, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, block, 10.0)
        val lbl = Mat()
        val st = Mat()
        val cen = Mat()
        val n = Imgproc.connectedComponentsWithStats(m, lbl, st, cen, 8, CvType.CV_32S)
        m.release(); cen.release()
        if (n < 2) {
            lbl.release(); st.release(); return null
        }
        val stats = IntArray(n * 5).also { st.get(0, 0, it) }
        st.release()
        val labels = IntArray(h * w).also { lbl.get(0, 0, it) }
        lbl.release()
        val hsOk = ArrayList<Float>()
        for (i in 1 until n) {
            val hh = stats[i * 5 + 3]
            val ww = stats[i * 5 + 2]
            if (hh >= 4 && hh < h * 0.05 && ww < w * 0.1) hsOk.add(hh.toFloat())
        }
        if (hsOk.size < 30) return null
        val ch = percentile(hsOk.toFloatArray(), 50.0).toDouble()
        // 글자(흐려서 글자끼리 붙은 덩어리 포함): 높이는 글자 크기, 폭은 글자 몇 개까지. 괘선·상자·그림·망점은 제외
        val keep = BooleanArray(n) { i ->
            if (i == 0) false else {
                val hh = stats[i * 5 + 3]; val ww = stats[i * 5 + 2]; val ar = stats[i * 5 + 4]
                hh >= 0.5 * ch && hh <= 2.5 * ch && ww >= 0.2 * ch && ww <= 8 * ch && ar >= 0.1 * ch * ch
            }
        }
        val gm = BooleanArray(h * w) { keep[labels[it]] && (chromaS == null || chromaS[it] <= CHROMA_KEEP) }
        var cntAll = 0
        for (v in gm) if (v) cntAll++
        if (cntAll < LOCAL_MIN_PIX * 4) return null
        val allPix = FloatArray(cntAll)
        var q = 0
        for (i in gm.indices) if (gm[i]) allPix[q++] = (gs[i].toInt() and 0xFF).toFloat()
        val ref = percentile(allPix, LOCAL_PCT).toDouble()
        val cell = max(8, (w * LOCAL_CELL).roundToInt())
        val gh = (h + cell - 1) / cell
        val gw = (w + cell - 1) / cell
        val valA = FloatArray(gh * gw)
        val okA = FloatArray(gh * gw)
        val glA = FloatArray(gh * gw)
        val buf = FloatArray(cell * cell)
        for (r in 0 until gh) for (c in 0 until gw) {
            val y0 = r * cell; val y1 = min(h, y0 + cell); val x0 = c * cell; val x1 = min(w, x0 + cell)
            var cnt = 0
            var gsum = 0.0
            var area = 0
            for (y in y0 until y1) for (x in x0 until x1) {
                val i = y * w + x
                area++
                if (glare != null) gsum += glare[i]
                if (gm[i]) buf[cnt++] = (gs[i].toInt() and 0xFF).toFloat()
            }
            val gl = if (glare != null && area > 0) (gsum / area).toFloat() else 0f
            glA[r * gw + c] = gl
            if (cnt >= LOCAL_MIN_PIX) {
                val v = percentile(buf.copyOf(cnt), LOCAL_PCT).toDouble()
                if (v <= paper * (LOCAL_CORE_MAX + (LOCAL_CORE_MAX_GLARE - LOCAL_CORE_MAX) * gl)) {
                    valA[r * gw + c] = v.toFloat(); okA[r * gw + c] = 1f
                }
            }
        }
        var okSum = 0f
        for (v in okA) okSum += v
        if (okSum < 1f) return null
        // 빈 칸은 이웃에서 메우며 매끈하게 (정규화 컨볼루션), 멀리 떨어진 빈 칸은 전체 기준값
        val num = Mat(gh, gw, CvType.CV_32F).also { it.put(0, 0, FloatArray(gh * gw) { valA[it] * okA[it] }) }
        val den = Mat(gh, gw, CvType.CV_32F).also { it.put(0, 0, okA) }
        val glM = Mat(gh, gw, CvType.CV_32F).also { it.put(0, 0, glA) }
        Imgproc.GaussianBlur(num, num, Size(0.0, 0.0), LOCAL_SIGMA, LOCAL_SIGMA, Core.BORDER_REPLICATE)
        Imgproc.GaussianBlur(den, den, Size(0.0, 0.0), LOCAL_SIGMA, LOCAL_SIGMA, Core.BORDER_REPLICATE)
        Imgproc.GaussianBlur(glM, glM, Size(0.0, 0.0), LOCAL_SIGMA, LOCAL_SIGMA, Core.BORDER_REPLICATE)
        val nA = FloatArray(gh * gw).also { num.get(0, 0, it) }
        val dA = FloatArray(gh * gw).also { den.get(0, 0, it) }
        val gbA = FloatArray(gh * gw).also { glM.get(0, 0, it) }
        val lvA = FloatArray(gh * gw) { i ->
            val d = dA[i]
            val v = if (d > 0.05f) nA[i] / max(d, 1e-6f) else ref.toFloat()
            val wgt = min(d / 0.3f, 1f)
            v * wgt + ref.toFloat() * (1f - wgt)
        }
        val smA = FloatArray(gh * gw) { (LOCAL_MAX_STRETCH + (LOCAL_MAX_STRETCH_GLARE - LOCAL_MAX_STRETCH) * gbA[it]).toFloat() }
        num.put(0, 0, lvA); den.put(0, 0, smA)
        val lvUp = Mat()
        val smUp = Mat()
        Imgproc.resize(num, lvUp, Size((gw * cell).toDouble(), (gh * cell).toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        Imgproc.resize(den, smUp, Size((gw * cell).toDouble(), (gh * cell).toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        listOf(num, den, glM).forEach { it.release() }
        val lv = FloatArray(h * w)
        val sm = FloatArray(h * w)
        val rowL = FloatArray(gw * cell)
        val rowS = FloatArray(gw * cell)
        for (y in 0 until h) {
            lvUp.get(y, 0, rowL); smUp.get(y, 0, rowS)
            System.arraycopy(rowL, 0, lv, y * w, w); System.arraycopy(rowS, 0, sm, y * w, w)
        }
        lvUp.release(); smUp.release()
        return LocalInk(lv, ref, sm)
    }

    /**
     * 조명 평탄화된 8비트 이미지 → 종이는 흰색, 글자는 원래 모양(가장자리의 부드러운 회색 포함) 그대로 진하게.
     * 글자를 이진화하듯 깎지 않는다(획이 끊기거나 얇아지거나 얼룩덜룩해지지 않게).
     *  - 흰색 기준을 자리마다 다르게: 진한 글자·긴 선·색 있는 곳 근처는 높게(가장자리 보존),
     *    그 밖은 낮게(뒷면에서 비치는 글씨와 종이 얼룩은 흰색으로)
     *  - 밝기만 바꾸고 색은 같은 비율로 유지(색 상자·그림 보존)
     * 입력이 RGB 면 RGB, 1채널이면 1채널(8비트)을 돌려준다. 메모리를 아끼려 큰 계산은 1채널로.
     */
    /**
     * bookscan.local_sharpness: 글자 가장자리 선명도 지도(칸 단위 Mat CV_32F, (gw*cell) x (gh*cell) 를 원본 크기로 자른 것).
     * 칸마다 글자 덩어리 가장자리 픽셀의 기울기(상위 25%)를 (종이 밝기 − 글자 검정)으로 나눈 값.
     * 선명하면 0.37 안팎, 흐리면 0.25 아래. 글자가 없는 칸은 이웃에서 메움. 반환: (지도, 글자 높이 px) 또는 (null, 0)
     */
    private fun localSharpness(g8: Mat, gF: Mat): Pair<Mat?, Double> {
        val h = g8.rows()
        val w = g8.cols()
        val block = max(15, (min(h, w) * 0.03).toInt()) or 1
        val m = Mat()
        Imgproc.adaptiveThreshold(g8, m, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, block, 12.0)
        val lbl = Mat()
        val st = Mat()
        val cen = Mat()
        val n = Imgproc.connectedComponentsWithStats(m, lbl, st, cen, 8, CvType.CV_32S)
        m.release(); cen.release()
        if (n < 2) {
            lbl.release(); st.release(); return Pair(null, 0.0)
        }
        val stats = IntArray(n * 5).also { st.get(0, 0, it) }
        st.release()
        val hsOk = ArrayList<Float>()
        for (i in 1 until n) {
            val hh = stats[i * 5 + 3]
            val ww = stats[i * 5 + 2]
            if (hh >= 4 && hh < h * 0.05 && ww < w * 0.1) hsOk.add(hh.toFloat())
        }
        if (hsOk.size < 30) {
            lbl.release(); return Pair(null, 0.0)
        }
        val ch = percentile(hsOk.toFloatArray(), 50.0).toDouble()
        val keep = BooleanArray(n) { i ->
            i != 0 && stats[i * 5 + 3] >= 0.5 * ch && stats[i * 5 + 3] <= 2.5 * ch && stats[i * 5 + 2] <= 8 * ch
        }
        val labels = IntArray(h * w).also { lbl.get(0, 0, it) }
        lbl.release()
        val gmB = ByteArray(h * w) { if (keep[labels[it]]) 1 else 0 }
        val gmM = Mat(h, w, CvType.CV_8UC1).also { it.put(0, 0, gmB) }
        val dil = Mat()
        val ero = Mat()
        val k3 = Mat.ones(3, 3, CvType.CV_8UC1)
        Imgproc.dilate(gmM, dil, k3)
        Imgproc.erode(gmM, ero, k3)
        k3.release(); gmM.release()
        val dB = ByteArray(h * w).also { dil.get(0, 0, it) }
        val eB = ByteArray(h * w).also { ero.get(0, 0, it) }
        dil.release(); ero.release()
        val gx = Mat()
        val gy = Mat()
        Imgproc.Sobel(gF, gx, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(gF, gy, CvType.CV_32F, 0, 1, 3)
        Core.magnitude(gx, gy, gx)
        gy.release()
        val mag = FloatArray(h * w).also { gx.get(0, 0, it) }
        gx.release()
        val gA = FloatArray(h * w).also { gF.get(0, 0, it) }
        val paper = percentile(gA, 60.0).toDouble()
        val cell = max(16, (w * SHARP_CELL).roundToInt())
        val gh = (h + cell - 1) / cell
        val gw = (w + cell - 1) / cell
        val valA = FloatArray(gh * gw)
        val okA = FloatArray(gh * gw)
        val edgeBuf = FloatArray(cell * cell)
        val coreBuf = FloatArray(cell * cell)
        for (r in 0 until gh) for (c in 0 until gw) {
            val y0 = r * cell; val y1 = min(h, y0 + cell); val x0 = c * cell; val x1 = min(w, x0 + cell)
            var ne = 0
            var ng = 0
            for (y in y0 until y1) for (x in x0 until x1) {
                val i = y * w + x
                if (gmB[i].toInt() != 0) coreBuf[ng++] = gA[i]
                if (dB[i].toInt() != 0 && eB[i].toInt() == 0) edgeBuf[ne++] = mag[i] / 8f
            }
            if (ne < SHARP_MIN_EDGE || ng == 0) continue
            val core = percentile(coreBuf.copyOf(ng), 10.0).toDouble()
            if (paper - core < 30) continue
            valA[r * gw + c] = (percentile(edgeBuf.copyOf(ne), 75.0) / (paper - core)).toFloat()
            okA[r * gw + c] = 1f
        }
        var okSum = 0f
        for (v in okA) okSum += v
        if (okSum < 1f) return Pair(null, 0.0)
        val valid = ArrayList<Float>()
        for (i in okA.indices) if (okA[i] > 0) valid.add(valA[i])
        val ref = percentile(valid.toFloatArray(), 50.0)
        val num = Mat(gh, gw, CvType.CV_32F).also { it.put(0, 0, FloatArray(gh * gw) { valA[it] * okA[it] }) }
        val den = Mat(gh, gw, CvType.CV_32F).also { it.put(0, 0, okA) }
        Imgproc.GaussianBlur(num, num, Size(0.0, 0.0), 1.0, 1.0, Core.BORDER_REPLICATE)
        Imgproc.GaussianBlur(den, den, Size(0.0, 0.0), 1.0, 1.0, Core.BORDER_REPLICATE)
        val nA = FloatArray(gh * gw).also { num.get(0, 0, it) }
        val dA = FloatArray(gh * gw).also { den.get(0, 0, it) }
        den.release()
        val smA = FloatArray(gh * gw) { i ->
            val d = dA[i]
            val v = if (d > 0.05f) nA[i] / max(d, 1e-6f) else ref
            val wgt = min(d / 0.3f, 1f)
            v * wgt + ref * (1f - wgt)
        }
        num.put(0, 0, smA)
        val up = Mat()
        Imgproc.resize(num, up, Size((gw * cell).toDouble(), (gh * cell).toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        num.release()
        val out = up.cropClone(0, h, 0, w)
        up.release()
        return Pair(out, ch)
    }

    /**
     * bookscan.restore_strokes: 흐려진 획 되살리기. g(CV_32F, 글자 진하기 맞춤이 끝난 밝기)를 받아 새 Mat 을 돌려준다
     * (바꿀 것이 없으면 g 그대로). loc/kA: 자리마다 글자 검정(lv)과 그 배율(1000px 축척), ink: 글자 후보(8U),
     * chPx: 글자 높이(0 이면 ink 에서 잼), lwF: 긴 선 가중치(0~1), colored: 색 가중치(0~1, 없으면 null), r: 글자 근처 반경
     */
    private fun restoreStrokes(
        g: Mat, paper: Double, black: Double, loc: LocalInk?, kA: FloatArray?, gsW: Int, gsH: Int,
        ink: Mat, chPx: Double, lwF: Mat, colored: Mat?, r: Int,
    ): Mat {
        val h = g.rows()
        val w = g.cols()
        val lbl = Mat()
        val st = Mat()
        val cen = Mat()
        val n = Imgproc.connectedComponentsWithStats(ink, lbl, st, cen, 8, CvType.CV_32S)
        cen.release()
        if (n < 2) {
            lbl.release(); st.release(); return g
        }
        val stats = IntArray(n * 5).also { st.get(0, 0, it) }
        st.release()
        var ch = chPx
        if (ch <= 0.0) {
            val hs = ArrayList<Float>()
            for (i in 1 until n) {
                val hh = stats[i * 5 + 3]
                if (hh >= 4 && hh < h * 0.05 && stats[i * 5 + 2] < w * 0.1) hs.add(hh.toFloat())
            }
            if (hs.size < 30) {
                lbl.release(); return g
            }
            ch = percentile(hs.toFloatArray(), 50.0).toDouble()
        }
        // 글자 크기 덩어리(그림·큰 상자·긴 선은 제외) 근처만
        val keep = BooleanArray(n) { i ->
            i != 0 && stats[i * 5 + 3] >= 2 && stats[i * 5 + 3] <= 3.0 * ch && stats[i * 5 + 2] <= 10.0 * ch
        }
        var nk = 0
        for (b in keep) if (b) nk++
        if (nk < 10) {
            lbl.release(); return g
        }
        val labels = IntArray(h * w).also { lbl.get(0, 0, it) }
        lbl.release()
        val tx8 = Mat(h, w, CvType.CV_8UC1).also { m -> m.put(0, 0, ByteArray(h * w) { if (keep[labels[it]]) 255.toByte() else 0 }) }
        Imgproc.dilate(tx8, tx8, ellipse(2 * r + 1))
        val txw = Mat()
        tx8.convertTo(txw, CvType.CV_32F, 1.0 / 255.0)
        tx8.release()
        Imgproc.GaussianBlur(txw, txw, Size(0.0, 0.0), max(1.0, r / 2.0))
        val one = Mat()
        lwF.convertTo(one, CvType.CV_32F, -1.0, 1.0) // 1 - lw
        Core.multiply(txw, one, txw)
        if (colored != null) {
            colored.convertTo(one, CvType.CV_32F, -1.0, 1.0)
            Core.multiply(txw, one, txw)
        }
        one.release()
        // 자리마다 글자 검정(진하기 맞춤 뒤): lv' = paper - (paper - lv) * k, 최대 paper - 30
        var lvFull = Mat()
        if (loc != null && kA != null) {
            val lvS = Mat(gsH, gsW, CvType.CV_32F).also { it.put(0, 0, FloatArray(loc.lv.size) { i ->
                min(paper - (paper - loc.lv[i]) * kA[i], paper - 30.0).toFloat()
            }) }
            Imgproc.resize(lvS, lvFull, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            lvS.release()
        } else {
            lvFull = Mat(h, w, CvType.CV_32F, Scalar(min(black, paper - 30.0)))
        }
        val sd = (STROKE_SMOOTH_CH * ch).coerceIn(0.6, 2.0)
        val gd = Mat()
        Imgproc.bilateralFilter(g, gd, 0, STROKE_SMOOTH_COLOR, sd)
        // t = clip((gd - lv') / max(paper - lv', 1), 0, 1)
        val den = Mat()
        lvFull.convertTo(den, CvType.CV_32F, -1.0, paper)
        Core.max(den, Scalar(1.0), den)
        val t = Mat()
        Core.subtract(gd, lvFull, t)
        gd.release()
        Core.divide(t, den, t)
        Core.max(t, Scalar(0.0), t)
        Core.min(t, Scalar(1.0), t)
        // z = (σ((t - mid)·a) - z0) / (z1 - z0),  σ(x) = 1/(1+e^-x)
        val a = STROKE_SLOPE * 4.0
        val z0 = 1.0 / (1.0 + exp(STROKE_MID * a))
        val z1 = 1.0 / (1.0 + exp(-(1.0 - STROKE_MID) * a))
        val z = Mat()
        t.convertTo(z, CvType.CV_32F, -a, STROKE_MID * a) // -(t - mid)·a
        t.release()
        Core.exp(z, z)
        Core.add(z, Scalar(1.0), z)
        Core.divide(1.0, z, z)
        z.convertTo(z, CvType.CV_32F, 1.0 / (z1 - z0), -z0 / (z1 - z0))
        Core.max(z, Scalar(0.0), z)
        Core.min(z, Scalar(1.0), z)
        // gr = lv' + (paper - lv') * z ;  out = g + (gr - g) * txw
        Core.multiply(den, z, den)
        z.release()
        Core.add(den, lvFull, den)
        lvFull.release()
        Core.subtract(den, g, den)
        Core.multiply(den, txw, den)
        txw.release()
        val out = Mat()
        Core.add(g, den, out)
        den.release()
        return out
    }

    private fun toneMap(norm: Mat, bgSmall: Mat? = null): Mat {
        val color = norm.channels() == 3
        val h = norm.rows()
        val w = norm.cols()
        var g8 = Mat()
        if (color) Imgproc.cvtColor(norm, g8, Imgproc.COLOR_RGB2GRAY) else norm.copyTo(g8)
        val gF0 = Mat()
        g8.convertTo(gF0, CvType.CV_32F) // 원래 밝기 (색 처리에서 L + (y - g) 의 g)
        // ---- 초점이 흐린 곳 선명하게: 글자 가장자리의 기울기(선명도)를 자리마다 재서, 흐린 곳일수록 강하게 언샤프
        //      (흐려진 획은 넓고 옅다. 대비만 키우면 굵고 뭉개진 글자가 되므로 먼저 획을 좁혀 세운다)
        var chPx = 0.0 // 글자 높이(px). 0 이면 못 잼
        run {
            val (shM, chPx0) = localSharpness(g8, gF0)
            chPx = chPx0
            if (shM != null) {
                val amt = Mat()
                Imgproc.resize(shM, amt, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
                shM.release()
                amt.convertTo(amt, CvType.CV_32F, -1.0 / SHARP_SPAN, SHARP_REF / SHARP_SPAN)
                Core.max(amt, Scalar(0.0), amt)
                Core.min(amt, Scalar(1.0), amt)
                Core.multiply(amt, Scalar(PRE_SHARPEN_MAX), amt)
                val mm = Core.minMaxLoc(amt)
                if (mm.maxVal > 0.05) {
                    val sig = (PRE_SHARPEN_SIGMA_CH * chPx).coerceIn(1.0, 2.5)
                    val gS = Mat()
                    Imgproc.GaussianBlur(gF0, gS, Size(0.0, 0.0), sig)
                    Core.subtract(gF0, gS, gS)
                    Core.multiply(gS, amt, gS)
                    Core.add(gF0, gS, gS)
                    Core.max(gS, Scalar(0.0), gS)
                    Core.min(gS, Scalar(255.0), gS)
                    g8.release()
                    g8 = Mat()
                    gS.convertTo(g8, CvType.CV_8U)
                    gS.release()
                }
                amt.release()
            }
        }
        // 종이 밝기·검정 기준 (긴 변 1000px)
        val s = min(1.0, 1000.0 / max(h, w))
        val gsM = Mat()
        if (s < 1.0) Imgproc.resize(g8, gsM, Size(0.0, 0.0), s, s, Imgproc.INTER_AREA) else g8.copyTo(gsM)
        val gs = ByteArray((gsM.total()).toInt()).also { gsM.get(0, 0, it) }
        val gsH = gsM.rows()
        val gsW = gsM.cols()
        val paper = percentileByHistogram(gs, PAPER_PCT)
        val coreThr = paper * INK_DARK
        val core = gs.filter { (it.toInt() and 0xFF) < coreThr }.toByteArray()
        var black = if (core.size > 100) percentileByHistogram(core, BLACK_INK_PCT) else percentileByHistogram(gs, 1.0)
        black = min(black, paper * WHITE_BG - 40.0)

        // 색: 종이 색(누런 기) 기준값 a_p, b_p (글자 진하기 지도에서 색 글자를 빼는 데도 씀)
        var lab: Mat? = null
        var aP = 128.0
        var bP = 128.0
        var chromaS: FloatArray? = null
        if (color) {
            val lb = Mat()
            Imgproc.cvtColor(norm, lb, Imgproc.COLOR_RGB2Lab)
            lab = lb
            val labS = Mat()
            if (s < 1.0) Imgproc.resize(lb, labS, Size(0.0, 0.0), s, s, Imgproc.INTER_AREA) else lb.copyTo(labS)
            val lsb = ByteArray((labS.total() * 3).toInt()).also { labS.get(0, 0, it) }
            labS.release()
            val ppThr = paper * 0.97
            val av = ArrayList<Byte>()
            val bv = ArrayList<Byte>()
            for (i in gs.indices) if ((gs[i].toInt() and 0xFF) >= ppThr) {
                av.add(lsb[i * 3 + 1]); bv.add(lsb[i * 3 + 2])
            }
            if (av.isNotEmpty()) {
                aP = percentileByHistogram(av.toByteArray(), 50.0)
                bP = percentileByHistogram(bv.toByteArray(), 50.0)
            }
            chromaS = FloatArray(gs.size) { i ->
                val da = (lsb[i * 3 + 1].toInt() and 0xFF) - aP
                val db = (lsb[i * 3 + 2].toInt() and 0xFF) - bP
                sqrt(da * da + db * db).toFloat()
            }
        }
        // ---- 글자 진하기 자리마다 맞춤: 옅게 찍힌 곳(초점 흐림·반사)의 글자 검정을 전체 글자 검정 자리로 끌어내림
        var glare: FloatArray? = null
        if (bgSmall != null) {
            val bgs = Mat()
            Imgproc.resize(bgSmall, bgs, gsM.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val ba = FloatArray((bgs.total()).toInt()).also { bgs.get(0, 0, it) }
            bgs.release()
            val med = max(percentile(ba, 50.0).toDouble(), 1.0)
            glare = FloatArray(ba.size) { ((ba[it] / med - 1.0 - GLARE_START) / (GLARE_FULL - GLARE_START)).coerceIn(0.0, 1.0).toFloat() }
        }
        val loc = localInkLevel(gsM, gs, paper, chromaS, glare)
        gsM.release()
        val gFs = Mat()
        g8.convertTo(gFs, CvType.CV_32F) // 선명하게 한 밝기 (글자 진하기 맞춤의 바탕)
        var gN = gFs
        var kAll: FloatArray? = null // 글자 진하기 배율(1000px 축척). 획 되살리기에서 자리마다 글자 검정을 구할 때 씀
        if (loc != null) {
            // 종이에서 떨어진 거리를 k 배: 그 칸의 글자 검정(lv)이 전체 글자 검정(ref) 자리로 온다. 옅은 곳만(k ≥ 1)
            val kA = FloatArray(loc.lv.size) { ((paper - loc.ref) / max(paper - loc.lv[it], 1.0)).coerceIn(1.0, loc.smax[it].toDouble()).toFloat() }
            kAll = kA
            var kMax = 1f
            for (v in kA) kMax = max(kMax, v)
            if (kMax > 1.02f) {
                val kS = Mat(gsH, gsW, CvType.CV_32F).also { it.put(0, 0, kA) }
                val kFull = Mat()
                Imgproc.resize(kS, kFull, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
                kS.release()
                gN = gFs.clone()
                // 많이 키우는 곳은 대개 초점이 흐린 곳: 획이 서로 붙지 않게 살짝 선명하게 한 뒤 키운다
                //   amt = clip((k-1)/2, 0, 1) * LOCAL_SHARPEN ;  g += (g - blur(g)) * amt
                val amt = Mat()
                kFull.convertTo(amt, CvType.CV_32F, 0.5, -0.5)
                Core.max(amt, Scalar(0.0), amt)
                Core.min(amt, Scalar(1.0), amt)
                Core.multiply(amt, Scalar(LOCAL_SHARPEN), amt)
                val bl = Mat()
                Imgproc.GaussianBlur(gN, bl, Size(0.0, 0.0), LOCAL_SHARPEN_SIGMA)
                Core.subtract(gN, bl, bl)
                Core.multiply(bl, amt, bl)
                Core.add(gN, bl, gN)
                bl.release(); amt.release()
                // g = paper - (paper - g) * k
                gN.convertTo(gN, CvType.CV_32F, -1.0, paper)
                Core.multiply(gN, kFull, gN)
                gN.convertTo(gN, CvType.CV_32F, -1.0, paper)
                Core.max(gN, Scalar(0.0), gN)
                Core.min(gN, Scalar(255.0), gN)
                kFull.release()
                g8.release()
                g8 = Mat()
                gN.convertTo(g8, CvType.CV_8U)
            }
        }

        // 글자 근처 지도: 진한 글자 심 주변 + 긴 선(상자 테두리·밑줄) + 색 있는 곳
        val ink = Mat()
        Imgproc.threshold(g8, ink, ceil(paper * INK_DARK) - 1.0, 255.0, Imgproc.THRESH_BINARY_INV) // g < paper*INK_DARK
        val faint = Mat()
        Imgproc.threshold(g8, faint, ceil(paper * LINE_FAINT) - 1.0, 255.0, Imgproc.THRESH_BINARY_INV)
        val ll = max(9, (w * 0.03).toInt())
        val lines = Mat()
        val tmp8 = Mat()
        Imgproc.morphologyEx(faint, lines, Imgproc.MORPH_OPEN, rectK(ll, 1))
        Imgproc.morphologyEx(faint, tmp8, Imgproc.MORPH_OPEN, rectK(1, ll))
        Core.bitwise_or(lines, tmp8, lines)
        faint.release()
        // 종이 접힌 자국·그림자 경계도 긴 '옅은 선'으로 잡히므로, 선 조각마다 충분히 진한 픽셀이 있어야 인쇄된 선으로 본다
        run {
            val lbl = Mat()
            val st = Mat()
            val cen = Mat()
            val nl = Imgproc.connectedComponentsWithStats(lines, lbl, st, cen, 8, CvType.CV_32S)
            st.release(); cen.release()
            if (nl > 1) {
                val labels = IntArray(w * h).also { lbl.get(0, 0, it) }
                val gb = ByteArray(w * h).also { g8.get(0, 0, it) }
                val cnt = IntArray(nl)
                val dark = IntArray(nl)
                val thr = paper * LINE_DARK
                for (i in 0 until w * h) {
                    val l = labels[i]
                    if (l == 0) continue
                    cnt[l]++
                    if ((gb[i].toInt() and 0xFF) < thr) dark[l]++
                }
                val keepL = BooleanArray(nl) { it != 0 && dark[it] >= LINE_DARK_FRAC * max(cnt[it], 1) }
                val lb = ByteArray(w * h) { if (keepL[labels[it]]) 255.toByte() else 0 }
                lines.put(0, 0, lb)
            }
            lbl.release()
        }
        val r = max(2, (w * INK_NEAR).toInt())
        val keep = Mat()
        Imgproc.dilate(ink, keep, ellipse(2 * r + 1))
        Imgproc.dilate(lines, tmp8, ellipse(5))
        Core.bitwise_or(keep, tmp8, keep)
        // 색: 종이 색(누런 기)을 뺀 채도. 색 상자·그림·색 글씨 근처는 지우지 않는다
        var chroma: Mat? = null
        if (color) {
            val lb = lab!!
            val ch = ArrayList<Mat>()
            Core.split(lb, ch)
            val aF = Mat()
            val bF = Mat()
            ch[1].convertTo(aF, CvType.CV_32F, 1.0, -aP)
            ch[2].convertTo(bF, CvType.CV_32F, 1.0, -bP)
            ch.forEach { it.release() }
            Core.magnitude(aF, bF, aF)
            bF.release()
            Imgproc.GaussianBlur(aF, aF, Size(0.0, 0.0), 2.0)
            chroma = aF
            val cm = Mat()
            Imgproc.threshold(aF, cm, CHROMA_KEEP, 255.0, Imgproc.THRESH_BINARY)
            cm.convertTo(tmp8, CvType.CV_8U)
            cm.release()
            Core.bitwise_or(keep, tmp8, keep)
        }
        // near(0~1) → 흰색 기준 white = paper*(WB + (WT-WB)*near) → white - black (최소 1)
        val wv = Mat()
        keep.convertTo(wv, CvType.CV_32F, 1.0 / 255.0)
        keep.release()
        Imgproc.GaussianBlur(wv, wv, Size(0.0, 0.0), max(1.0, r / 2.0))
        wv.convertTo(wv, CvType.CV_32F, paper * (WHITE_TEXT - WHITE_BG), paper * WHITE_BG - black)
        Core.max(wv, Scalar(1.0), wv)
        // 긴 선 가중치 lw = blur(dilate(lines)) (0~1): S자 곡선과 획 되살리기에서 선 위는 제외
        val lwF = Mat()
        Imgproc.dilate(lines, tmp8, rectK(3, 3))
        lines.release()
        tmp8.convertTo(lwF, CvType.CV_32F, 1.0 / 255.0)
        Imgproc.GaussianBlur(lwF, lwF, Size(0.0, 0.0), 1.0)
        // 색 처리용: colored = clip((chroma - K/2)/K, 0, 1) — 색이 있는 곳만 색을 남김
        // (검은 글자의 가장자리 색 번짐은 무채색으로). 순수한 흰색은 '색이 없는' 종이 부분만
        var fade: Mat? = null
        if (color) {
            val col = Mat()
            chroma!!.convertTo(col, CvType.CV_32F, 1.0 / CHROMA_KEEP, -0.5)
            Core.max(col, Scalar(0.0), col)
            Core.min(col, Scalar(1.0), col)
            fade = col
        }
        // ---- 획 모양 되살리기: 흐려진 획은 넓고 옅다. 잡음을 살짝 누른 뒤 자리마다 (글자 검정 ~ 종이) 한가운데를
        //      중심으로 가파른 S-곡선 → 가장자리가 원래 자리에서 서고 획 폭이 원래대로. 글자 크기 덩어리 근처에서만
        run {
            val gR = restoreStrokes(gN, paper, black, loc, kAll, gsW, gsH, ink, chPx, lwF, fade, r)
            if (gR !== gN) {
                if (gN !== gFs) gN.release()
                gN = gR
            }
        }
        ink.release()
        // t = clip((g - black) / (white - black), 0, 1)   (g 는 글자 진하기를 맞춘 것)
        g8.release()
        val t = Mat()
        Core.subtract(gN, Scalar(black), t)
        Core.divide(t, wv, t)
        wv.release()
        if (gN !== gFs) gN.release()
        gFs.release()
        val gF = gF0 // 원래 밝기 (색 처리에서 L + (y - g) 의 g)
        Core.max(t, Scalar(0.0), t)
        Core.min(t, Scalar(1.0), t)
        val white = Mat()
        Imgproc.threshold(t, white, 0.999999, 255.0, Imgproc.THRESH_BINARY) // t >= 1
        val white8 = Mat()
        white.convertTo(white8, CvType.CV_8U)
        white.release()
        if (color) {
            val colMask = Mat()
            Imgproc.threshold(chroma, colMask, CHROMA_KEEP * 0.5, 255.0, Imgproc.THRESH_BINARY)
            colMask.convertTo(tmp8, CvType.CV_8U)
            colMask.release()
            Core.subtract(white8, tmp8, white8) // (t >= 1) 이면서 색이 없는 곳
            chroma!!.release()
        }
        // 곡선: 감마(글자 진하게)와 S자 곡선 t²(3-2t)(글자 둘레의 옅은 얼룩·비침을 흰색 쪽으로)을 섞는다.
        // 상자 테두리·밑줄 같은 옅은 긴 선 위에서는 S자를 쓰지 않는다(옅은 선이 사라지지 않게).
        //   sm = S_MIX * (1 - blur(dilate(lines)))  ;  u = t + sm * (S(t) - t)
        val sm = Mat()
        lwF.convertTo(sm, CvType.CV_32F, -S_MIX, S_MIX)
        lwF.release()
        val u = Mat()
        t.convertTo(u, CvType.CV_32F, -2.0, 3.0)
        Core.multiply(u, t, u)
        Core.multiply(u, t, u)        // S(t)
        Core.subtract(u, t, u)        // S(t) - t
        Core.multiply(u, sm, u)
        Core.add(t, u, t)
        u.release(); sm.release()
        Core.pow(t, TONE_GAMMA, t)
        t.convertTo(t, CvType.CV_32F, 255.0 - BLACK_OUT, BLACK_OUT)
        val out = Mat()
        if (color) {
            // 밝기는 톤 곡선대로(L + (y - g)), 색(a, b)은 종이 색을 빼고 조금 진하게. 종이 부분은 순수한 흰색.
            // (RGB 에 같은 비율을 곱하면 밝은 색이 255 에서 잘려 색이 바래므로 Lab 에서 처리)
            val ch = ArrayList<Mat>()
            Core.split(lab!!, ch)
            lab.release()
            val lF = Mat()
            ch[0].convertTo(lF, CvType.CV_32F)
            Core.subtract(t, gF, t)
            Core.add(lF, t, lF)
            val outCh = ArrayList<Mat>()
            outCh.add(Mat().also { lF.convertTo(it, CvType.CV_8U) })
            lF.release()
            val k = COLOR_SAT
            for ((c, ref) in listOf(1 to aP, 2 to bP)) {
                val f = Mat()
                ch[c].convertTo(f, CvType.CV_32F, 1.0, -ref)
                Core.multiply(f, fade!!, f, k)
                val o8 = Mat()
                f.convertTo(o8, CvType.CV_8U, 1.0, 128.0)
                f.release()
                outCh.add(o8)
            }
            ch.forEach { it.release() }
            fade!!.release()
            val labOut = Mat()
            Core.merge(outCh, labOut)
            outCh.forEach { it.release() }
            Imgproc.cvtColor(labOut, out, Imgproc.COLOR_Lab2RGB)
            labOut.release()
            out.setTo(Scalar(255.0, 255.0, 255.0), white8)
        } else {
            t.convertTo(out, CvType.CV_8U)
        }
        gF.release(); t.release(); white8.release(); tmp8.release()
        if (SHARPEN > 0) {
            val bl = Mat()
            Imgproc.GaussianBlur(out, bl, Size(0.0, 0.0), SHARPEN_SIGMA)
            Core.addWeighted(out, 1.0 + SHARPEN, bl, -SHARPEN, 0.0, out)
            bl.release()
        }
        return out
    }

    private fun finish(norm: Mat, mode: ColorMode, bgSmall: Mat? = null): Mat {
        if (mode == ColorMode.COLOR) return toneMap(norm, bgSmall)
        val g = Mat()
        Imgproc.cvtColor(norm, g, Imgproc.COLOR_RGB2GRAY)
        val y = toneMap(g, bgSmall)
        g.release()
        if (mode == ColorMode.GRAY) return y
        // 흑백: 조명이 이미 고르게 펴졌으므로 한 기준으로 자른다(적응형 이진화는 흐린 획을 끊어 먹음).
        // 해상도가 낮으면 2배로 키워 자른 뒤 되돌려 글자 가장자리가 계단지지 않게.
        val w = y.cols()
        val h = y.rows()
        val bw = Mat()
        if (w < BW_UPSCALE_BELOW) {
            val up = Mat()
            y.convertTo(up, CvType.CV_32F)
            Imgproc.resize(up, up, Size(w * 2.0, h * 2.0), 0.0, 0.0, Imgproc.INTER_CUBIC)
            Imgproc.GaussianBlur(up, up, Size(0.0, 0.0), 1.0)
            Imgproc.threshold(up, up, BW_T - 1e-3, 255.0, Imgproc.THRESH_BINARY) // up >= BW_T → 흰색
            val up8 = Mat()
            up.convertTo(up8, CvType.CV_8U)
            up.release()
            Imgproc.resize(up8, bw, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            up8.release()
        } else {
            val yf = Mat()
            y.convertTo(yf, CvType.CV_32F)
            Imgproc.GaussianBlur(yf, yf, Size(0.0, 0.0), 0.7)
            Imgproc.threshold(yf, yf, BW_T - 1e-3, 255.0, Imgproc.THRESH_BINARY)
            yf.convertTo(bw, CvType.CV_8U)
            yf.release()
        }
        y.release()
        return bw
    }

    /**
     * 가장자리 띠(각 변의 4%, 접힌 선 쪽은 6%) 안의 선 찌꺼기(페이지 경계선, 접힌 선 그림자, 책 옆면 줄무늬)를 흰색으로.
     *  - 긴 세로선·가로선: 글자는 짧은 획들이라 긴 선 모양 걸러내기(열림 연산)를 통과하지 못한다.
     *  - 띠 안에 완전히 들어 있는 가늘고 긴 조각(좌우: 세로로 2.5배 이상, 위아래: 가로로 8배 이상·폭 5% 이상)
     *  - 좌우 변에 붙은 세로 잔무늬 띠(바닥 카펫·책 옆면 찌꺼기): 어두운 점이 빽빽한 곳이 변에 닿아 있고
     *    안쪽으로 얕게(8% 이내)만 들어왔으며 세로로 긴 것. 글줄(가로로 긴 덩어리)은 건드리지 않는다.
     * out 을 직접 수정. gutterSide: 접힌 선 쪽
     */
    private fun cleanEdges(out: Mat, gutterSide: Side?) {
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
        val bl = max(2, (sw * (if (gutterSide == Side.LEFT) EDGE_GUTTER else EDGE_BAND)).toInt())
        val br = max(2, (sw * (if (gutterSide == Side.RIGHT) EDGE_GUTTER else EDGE_BAND)).toInt())
        val by = max(2, (sh * EDGE_BAND).toInt())
        val kv = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, max(5, (sh * 0.04).toInt()).toDouble()))
        val kh = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(max(5, (sw * 0.04).toInt()).toDouble(), 1.0))
        val mb = ByteArray(sh * sw)
        var any = false
        // 띠 안의 긴 선 조각마다: 상자 테두리(가로로 넓게 이어진 덩어리의 일부이고 페이지 위아래 끝에 닿지 않음)는 남기고,
        // 페이지 경계선·접힌 곳 그림자(홀로 떨어진 선이거나 페이지 끝까지 이어진 선)는 지운다.
        // 접힌 선 쪽 띠에서는 옅은 그림자 선(밝기 EDGE_FAINT 미만)까지 본다.
        class Prep(val vert: ByteArray, val horz: ByteArray, val wide: BooleanArray, val tall: BooleanArray, val labels: IntArray)
        val cache = HashMap<Int, Prep>()
        fun prep(thr: Int): Prep = cache.getOrPut(thr) {
            val d = Mat()
            Imgproc.threshold(g, d, thr - 1.0, 255.0, Imgproc.THRESH_BINARY_INV) // g < thr
            val v = Mat()
            val hz = Mat()
            Imgproc.morphologyEx(d, v, Imgproc.MORPH_OPEN, kv)
            Imgproc.morphologyEx(d, hz, Imgproc.MORPH_OPEN, kh)
            val lbl = Mat()
            val st = Mat()
            val cen = Mat()
            val n = Imgproc.connectedComponentsWithStats(d, lbl, st, cen, 8, CvType.CV_32S)
            val stats = IntArray(n * 5).also { st.get(0, 0, it) }
            val labels = IntArray(sh * sw).also { lbl.get(0, 0, it) }
            val p = Prep(
                ByteArray(sh * sw).also { v.get(0, 0, it) }, ByteArray(sh * sw).also { hz.get(0, 0, it) },
                BooleanArray(n) { stats[it * 5 + 2] > sw * EDGE_BOX }, BooleanArray(n) { stats[it * 5 + 3] > sh * EDGE_BOX }, labels
            )
            listOf(d, v, hz, lbl, st, cen).forEach { it.release() }
            p
        }
        // (이름, 띠 x0, x1, y0, y1, 세로선인가)
        val bands = listOf(
            Triple(Side.LEFT, intArrayOf(0, bl, 0, sh), true), Triple(Side.RIGHT, intArrayOf(sw - br, sw, 0, sh), true),
            Triple(null, intArrayOf(0, sw, 0, by), false), Triple(null, intArrayOf(0, sw, sh - by, sh), false)
        )
        for ((side, r, vertical) in bands) {
            val pr = prep(if (side != null && side == gutterSide) EDGE_FAINT else 200)
            val src = if (vertical) pr.vert else pr.horz
            val segB = ByteArray(sh * sw)
            for (y in r[2] until r[3]) for (x in r[0] until r[1]) segB[y * sw + x] = src[y * sw + x]
            val segM = Mat(sh, sw, CvType.CV_8UC1).also { it.put(0, 0, segB) }
            val sl = Mat()
            val ss = Mat()
            val sc = Mat()
            val ns = Imgproc.connectedComponentsWithStats(segM, sl, ss, sc, 8, CvType.CV_32S)
            val sst = IntArray(ns * 5).also { ss.get(0, 0, it) }
            val slab = IntArray(sh * sw).also { sl.get(0, 0, it) }
            listOf(segM, sl, ss, sc).forEach { it.release() }
            val alongLo = if (vertical) by else bl
            val alongHi = if (vertical) sh - by else sw - br
            for (j in 1 until ns) {
                val x0 = sst[j * 5]
                val y0 = sst[j * 5 + 1]
                val bw = sst[j * 5 + 2]
                val bh = sst[j * 5 + 3]
                var pix = 0
                var big = 0
                for (y in y0 until y0 + bh) for (x in x0 until x0 + bw) {
                    val k = y * sw + x
                    if (slab[k] != j) continue
                    pix++
                    val o = pr.labels[k]
                    if (if (vertical) pr.wide[o] else pr.tall[o]) big++
                }
                val boxy = big > 0.5 * pix
                val a0 = if (vertical) y0 else x0
                val a1 = if (vertical) y0 + bh else x0 + bw
                val reachesEnd = a0 <= alongLo || a1 >= alongHi
                // 접힌 선 쪽: 페이지 높이의 절반 넘게 이어진 세로선은 상자 테두리에 닿아 있어도 접힌 곳 그림자다
                val fold = side != null && side == gutterSide && (a1 - a0) >= EDGE_FOLD_LEN * sh
                if (boxy && !reachesEnd && !fold) continue
                for (y in y0 until y0 + bh) for (x in x0 until x0 + bw) {
                    val k = y * sw + x
                    if (slab[k] == j) {
                        mb[k] = 255.toByte(); any = true
                    }
                }
            }
        }
        kv.release(); kh.release(); g.release()
        // 띠 안의 가늘고 긴 조각
        run {
            val lbl = Mat()
            val st = Mat()
            val cen = Mat()
            val n = Imgproc.connectedComponentsWithStats(dark, lbl, st, cen, 8, CvType.CV_32S)
            val stats = IntArray(n * 5).also { st.get(0, 0, it) }
            val labels = IntArray(sh * sw).also { lbl.get(0, 0, it) }
            lbl.release(); st.release(); cen.release()
            val hit = BooleanArray(n)
            for (i in 1 until n) {
                val x = stats[i * 5]
                val y = stats[i * 5 + 1]
                val bw = stats[i * 5 + 2]
                val bh = stats[i * 5 + 3]
                // 글자 획 하나(예: 'ㅣ')를 지우지 않도록 글자 높이보다 훨씬 긴 것만
                val sideLr = (x + bw <= bl || x >= sw - br) && bh >= 2.5 * bw && bh >= sh * EDGE_BOX
                // 위아래는 아주 긴 가로 조각(페이지 경계선)만: 글자 획 하나(예: '그'의 'ㅡ')를 지우지 않게
                val sideTb = (y + bh <= by || y >= sh - by) && bw >= 8 * bh && bw >= sw * 0.05
                // 변에 닿은 가늘고 짧은 획(책 옆면 종이 단면·바닥 줄이 페이지 끝까지 이어진 것). 글자는 변에 닿지 않는다
                val touch = x == 0 || y == 0 || x + bw == sw || y + bh == sh
                val long = max(bw, bh)
                val short = min(bw, bh)
                val stroke = touch && long >= 2.5 * short && short <= sw * EDGE_STROKE_THIN && long <= sh * EDGE_STROKE_MAX
                hit[i] = sideLr || sideTb || stroke
            }
            for (k in 0 until sh * sw) if (hit[labels[k]]) {
                mb[k] = 255.toByte(); any = true
            }
        }
        // 변에 붙은 잔무늬 덩어리
        run {
            val kk = max(5, (min(sh, sw) * 0.015).toInt()) or 1
            val darkF = Mat()
            dark.convertTo(darkF, CvType.CV_32F, 1.0 / 255.0)
            val dens = Mat()
            Imgproc.boxFilter(darkF, dens, -1, Size(kk.toDouble(), kk.toDouble()))
            darkF.release()
            val tex = Mat()
            Imgproc.threshold(dens, tex, 0.10, 255.0, Imgproc.THRESH_BINARY)
            dens.release()
            val tex8 = Mat()
            tex.convertTo(tex8, CvType.CV_8U)
            tex.release()
            val lbl = Mat()
            val st = Mat()
            val cen = Mat()
            val n = Imgproc.connectedComponentsWithStats(tex8, lbl, st, cen, 8, CvType.CV_32S)
            tex8.release(); cen.release()
            val stats = IntArray(n * 5).also { st.get(0, 0, it) }
            val labels = IntArray(sh * sw).also { lbl.get(0, 0, it) }
            lbl.release(); st.release()
            val tx = sw * 0.01
            val dx = sw * EDGE_TEXTURE
            val hit = BooleanArray(n)
            for (i in 1 until n) {
                val x = stats[i * 5]
                val bw = stats[i * 5 + 2]
                val bh = stats[i * 5 + 3]
                // 좌우 변에 붙어 있고, 안쪽으로 얕고, 세로로 긴 것만 (글줄은 가로로 길어 해당 없음)
                hit[i] = ((x <= tx && x + bw <= dx) || (x + bw >= sw - tx && x >= sw - dx)) && bh >= 3 * bw
            }
            for (k in 0 until sh * sw) if (hit[labels[k]]) {
                mb[k] = 255.toByte(); any = true
            }
        }
        // 접힌 선 쪽 변에 붙은 찌꺼기(옆 페이지 글자의 시작이 접힌 곳에서 가로로 늘어난 얼룩 등): 변에 닿은 어두운 것이
        // 띠 안에서 깨끗한 흰 틈(폭 1% 이상)으로 본문과 떨어져 있으면 그 틈까지 통째로 지운다. 안쪽 여백은 늘 있으므로 안전
        if (gutterSide == Side.LEFT || gutterSide == Side.RIGHT) {
            val db = ByteArray(sh * sw).also { dark.get(0, 0, it) }
            val inkc = FloatArray(sw)
            for (y in 0 until sh) for (x in 0 until sw) if (db[y * sw + x].toInt() != 0) inkc[x] += 1f
            for (x in 0 until sw) inkc[x] /= sh.toFloat()
            val right = gutterSide == Side.RIGHT
            fun col(i: Int): Float = if (right) inkc[sw - 1 - i] else inkc[i]
            val gap = max(3, (sw * EDGE_REM_GAP).toInt())
            val bandw = if (right) br else bl
            var content = false
            for (i in 0 until gap) if (col(i) > EDGE_REM_INK) content = true
            if (content) {
                var run = 0
                var cut = -1
                for (i in 0 until min(sw, bandw + gap)) {
                    run = if (col(i) <= EDGE_REM_INK) run + 1 else 0
                    if (run >= gap) {
                        cut = i - gap + 1; break
                    }
                }
                if (cut in 1..bandw) {
                    for (y in 0 until sh) for (i in 0 until cut) {
                        mb[y * sw + (if (right) sw - 1 - i else i)] = 255.toByte()
                    }
                    any = true
                }
            }
        }
        dark.release()
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

    /**
     * 손가락 영역을 흰색으로 채우고(경계는 부드럽게), 주변 띠에 남은 옅은 그림자·테두리도 흰색으로.
     * 띠에서는 ERASE_LIGHT 보다 밝은 것만 밀어내므로 글자(가장자리의 회색 포함)는 남는다. out 을 직접 수정.
     */
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
        val roiS = expand(union(Imgproc.boundingRect(ringS), toSmall(Imgproc.boundingRect(mask), s)), max(6, (18 * s).toInt() + 1), ss)
        Imgproc.GaussianBlur(ringS, ringS, Size(0.0, 0.0), max(1.0, 6.0 * s))
        val roiF = toFull(roiS, s, w, h)

        val ring = Mat()
        ringS.withRoi(roiS) { Imgproc.resize(it, ring, roiF.size(), 0.0, 0.0, Imgproc.INTER_LINEAR) }
        ring.convertTo(ring, CvType.CV_32F, 1.0 / 255.0)
        val soft = Mat()
        mask.withRoi(roiF) { Imgproc.GaussianBlur(it, soft, Size(0.0, 0.0), 3.0) }
        soft.convertTo(soft, CvType.CV_32F, 1.0 / 255.0)

        val tmp = Mat()
        val f = Mat()
        out.withRoi(roiF) { outRoi ->
            val lum = Mat()
            if (out.channels() == 3) Imgproc.cvtColor(outRoi, lum, Imgproc.COLOR_RGB2GRAY) else outRoi.copyTo(lum)
            // light = clip((lum - ERASE_LIGHT) / 40, 0, 1) * ring ;  wgt = max(soft, light)
            lum.convertTo(lum, CvType.CV_32F, 1.0 / 40.0, -ERASE_LIGHT / 40.0)
            Core.max(lum, Scalar(0.0), lum)
            Core.min(lum, Scalar(1.0), lum)
            Core.multiply(lum, ring, lum)
            Core.max(soft, lum, lum)
            // out = out + (255 - out) * wgt, 채널 하나씩
            for (c in 0 until out.channels()) {
                if (out.channels() == 3) Core.extractChannel(outRoi, tmp, c) else outRoi.copyTo(tmp)
                tmp.convertTo(f, CvType.CV_32F)
                f.convertTo(f, CvType.CV_32F, -1.0, 255.0)   // 255 - out
                Core.multiply(f, lum, f)
                val o = Mat()
                tmp.convertTo(o, CvType.CV_32F)
                Core.add(o, f, o)
                o.convertTo(tmp, CvType.CV_8U)
                o.release()
                if (out.channels() == 3) Core.insertChannel(tmp, outRoi, c) else tmp.copyTo(outRoi)
            }
            lum.release()
        }
        listOf(mS, ringS, ring, soft, tmp, f).forEach { it.release() }
    }

    // ================================================================= 유틸

    private fun ellipse(k: Int): Mat = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))

    private fun rectK(w: Int, h: Int): Mat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(w.toDouble(), h.toDouble()))

    /** 부분 영역의 독립 복사본. submat 헤더를 바로 해제해 큰 부모 버퍼가 붙잡히지 않게 */
    private fun Mat.cropClone(r: Rect): Mat {
        val v = submat(r)
        try {
            return v.clone()
        } finally {
            v.release()
        }
    }

    private fun Mat.cropClone(r0: Int, r1: Int, c0: Int, c1: Int): Mat = cropClone(Rect(c0, r0, c1 - c0, r1 - r0))

    /** 부분 영역 보기(같은 버퍼)로 작업한 뒤 보기 헤더를 해제 */
    private inline fun <T> Mat.withRoi(r: Rect, f: (Mat) -> T): T {
        val v = submat(r)
        try {
            return f(v)
        } finally {
            v.release()
        }
    }

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

    /** 원본 좌표 상자 → 축소(배율 s) 좌표 상자(바깥으로 반올림) */
    private fun toSmall(r: Rect, s: Double): Rect {
        val x0 = floor(r.x * s).toInt()
        val y0 = floor(r.y * s).toInt()
        val x1 = ceil((r.x + r.width) * s).toInt()
        val y1 = ceil((r.y + r.height) * s).toInt()
        return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))
    }

    /** 두 상자를 감싸는 상자 (크기 0인 상자는 무시) */
    private fun union(a: Rect, b: Rect): Rect {
        if (b.width <= 0 || b.height <= 0) return a
        if (a.width <= 0 || a.height <= 0) return b
        val x0 = min(a.x, b.x)
        val y0 = min(a.y, b.y)
        val x1 = max(a.x + a.width, b.x + b.width)
        val y1 = max(a.y + a.height, b.y + b.height)
        return Rect(x0, y0, x1 - x0, y1 - y0)
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

        // ---- 손가락: 변에서 들어온 깊이 최소(변과 수직 길이 대비), 몸통이 변을 따라 차지하는 길이 최대(변 길이 대비),
        //      몸통의 (변을 따라 길이)/(깊이) 최대 — 이보다 길쭉하면 책 옆면 띠
        const val FINGER_MIN_DEPTH = 0.02
        const val FINGER_MAX_ALONG = 0.40
        const val FINGER_ALONG_DEPTH = 2.0

        // ---- 화질 보정(톤): engine/bookscan.py 의 같은 이름 상수
        const val PAPER_PCT = 60.0      // 종이 밝기 = 조명 평탄화 후 밝기의 이 백분위
        const val WHITE_TEXT = 0.95     // 글자 근처의 흰색 기준(종이 밝기 대비): 글자 가장자리의 부드러운 회색을 살림
        const val WHITE_BG = 0.86       // 글자에서 떨어진 곳의 흰색 기준: 뒷면 비침·종이 결을 지움
        const val INK_DARK = 0.62       // 종이 밝기의 이 배보다 어두우면 확실한 글자(인쇄 잉크)
        const val LINE_FAINT = 0.95     // 상자 테두리·밑줄처럼 긴 선은 옅어도 살림
        const val LINE_DARK = 0.80      // 긴 선 조각 중 이 비율보다 어두운 픽셀이
        const val LINE_DARK_FRAC = 0.25 // 이만큼은 있어야 인쇄된 선(접힌 자국·그림자 경계는 더 옅고 고름)
        const val INK_NEAR = 0.006      // 글자 근처로 보는 거리(페이지 폭 비율)
        const val CHROMA_KEEP = 8.0     // 종이 색을 뺀 채도가 이보다 진하면(색 상자·그림·색 글씨) 살림
        // 초점 흐린 곳 선명하게 (engine/bookscan.py SHARP_* / PRE_SHARPEN_* 와 같은 값)
        const val SHARP_REF = 0.37
        const val SHARP_SPAN = 0.13
        const val PRE_SHARPEN_MAX = 2.5
        const val PRE_SHARPEN_SIGMA_CH = 0.1
        const val SHARP_CELL = 1.0 / 10
        const val SHARP_MIN_EDGE = 300
        // 획 모양 되살리기 (engine/bookscan.py STROKE_* 와 같은 값)
        const val STROKE_SLOPE = 3.0
        const val STROKE_MID = 0.5
        const val STROKE_SMOOTH_CH = 0.09
        const val STROKE_SMOOTH_COLOR = 20.0
        // 글자 진하기 자리마다 맞춤 (engine/bookscan.py LOCAL_* / GLARE_* 와 같은 값)
        const val LOCAL_CELL = 1.0 / 12
        const val LOCAL_PCT = 5.0
        const val LOCAL_MIN_PIX = 60
        const val LOCAL_CORE_MAX = 0.72
        const val LOCAL_MAX_STRETCH = 3.0
        const val LOCAL_SIGMA = 1.0
        const val GLARE_START = 0.06
        const val GLARE_FULL = 0.18
        const val LOCAL_CORE_MAX_GLARE = 0.95
        const val LOCAL_MAX_STRETCH_GLARE = 8.0
        const val LOCAL_SHARPEN = 1.0
        const val LOCAL_SHARPEN_SIGMA = 2.0
        const val COLOR_SAT = 1.3       // 색을 이만큼 진하게(1 = 그대로)
        const val BLACK_INK_PCT = 50.0  // 글자 픽셀 밝기의 이 백분위를 완전한 검정으로
        const val BLACK_OUT = 12.0      // 검정이 놓일 값 (0으로 몰면 글자 가장자리가 뭉개짐)
        const val TONE_GAMMA = 1.5      // 1보다 크면 글자가 더 진하고 굵게
        const val S_MIX = 0.6           // S자 곡선 섞는 비율(0~1): 클수록 글자 둘레 얼룩이 깨끗해지지만 옅은 인쇄도 옅어짐
        const val FLAT_PREBLUR = 1.5    // 조명 평탄화: 배경 추정 전 흐림(긴 변 1000px 기준)
        const val DENOISE_GAIN0 = 1.5     // 조명 평탄화 배율이 이보다 크면 흐리기 시작
        const val DENOISE_GAIN_SPAN = 2.5 // 배율 4(배경 밝기 64) 에서 완전히 흐림
        const val DENOISE_SIGMA = 1.0
        const val SHARPEN = 0.3         // 언샤프 마스크 세기 (너무 세면 글자가 얼룩덜룩해짐)
        const val SHARPEN_SIGMA = 1.0
        const val BW_T = 140.0          // 흑백 모드: 톤 보정 후 이보다 어두우면 검정
        const val BW_UPSCALE_BELOW = 1600 // 흑백 모드: 페이지 폭이 이보다 작으면 2배로 키워서 자름
        const val ERASE_LIGHT = 200.0   // 손가락 주변 띠에서 이보다 밝은 것(남은 옅은 그림자)만 흰색으로

        /** 글줄 분석 해상도(긴 변) */
        const val ANALYSIS_SIDE = 1600.0

        /** 모델 격자를 위아래로 연장할 칸 수 */
        const val GRID_EXT = 1
        const val GRID_EXT_H_MAX = 8      // 좌우로 연장할 최대 칸 수
        // 페이지 가장자리 띠 (engine/bookscan.py EDGE_PAPER_* / EDGE_BAND_* 와 같은 값)
        const val EDGE_PAPER_MIN = 0.95
        const val EDGE_PAPER_CHROMA = 2.5
        const val EDGE_BAND_DARK = 0.85
        const val EDGE_BAND_CHROMA = 4.0
        const val EDGE_BAND_TEXT = 0.02
        const val EDGE_BAND_MAX = 0.15
        const val EXT_STRADDLE_MIN = 6    // 경계선 양쪽에 걸친 글자 덩어리가 이보다 많아야 '글자를 가로지른 것'

        /** 한 쪽 페이지가 이보다 가로로 길면 세로로 눕혀서 곡면 펴기 */
        const val LANDSCAPE_RATIO = 1.2

        /** 가로세로 비율 보정: 국소 배율비의 이 백분위를 '평평한 곳'의 값으로 봄, 허용 범위 */
        const val ASPECT_PCT = 75.0
        const val ASPECT_MIN = 0.85
        const val ASPECT_MAX = 1.25

        /** 책 옆면 줄무늬 판정: 변과 수직 방향 기울기 합이 나란한 방향의 이 배수보다 크면 */
        const val STREAK_RATIO = 2.2

        /** 곡면 편 결과에서 책 마스크를 이 비율만큼 안쪽으로 줄여 씀(마스크 경계가 거침) */
        const val PAGE_MASK_ERODE = 0.012

        /** 곡면 펴기 모델 입력: 페이지 상자 바깥 여유, 마스크를 넓힐 비율, 바깥 칠할 색 */
        const val ISOLATE_MARGIN = 0.03
        const val ISOLATE_DILATE = 0.02
        const val ISOLATE_FILL = 30.0

        /** 가장자리 정리 띠 폭(각 변, 접힌 선 쪽), 변에 붙은 잔무늬 덩어리가 들어올 수 있는 깊이 */
        const val EDGE_BAND = 0.04
        const val EDGE_GUTTER = 0.06
        const val EDGE_TEXTURE = 0.08
        const val EDGE_REM_GAP = 0.01   // 접힌 선 쪽 변의 찌꺼기와 본문 사이 흰 틈의 최소 폭(페이지 폭 비율)
        const val EDGE_REM_INK = 0.002  // 열에 어두운 픽셀이 이 비율 이하면 '깨끗한' 열
        const val EDGE_FOLD_LEN = 0.5      // 접힌 선 쪽 띠에서 이 비율(페이지 높이) 넘게 이어진 세로선은 상자 테두리로 보지 않음
        const val EDGE_STROKE_THIN = 0.006 // 변에 닿은 획을 지울 최대 굵기(페이지 폭 비율)
        const val EDGE_STROKE_MAX = 0.10   // 변에 닿은 획을 지울 최대 길이(페이지 높이 비율)
        const val EDGE_FAINT = 245 // 접힌 선 쪽 띠에서 그림자 선으로 볼 밝기 기준(이보다 어두우면)
        const val EDGE_BOX = 0.03  // 가장자리 선이 이보다 넓게(세로선) / 높게(가로선) 이어진 덩어리에 속하면 상자 테두리로 보고 남김

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
