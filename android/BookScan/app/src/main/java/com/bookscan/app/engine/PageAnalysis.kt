package com.bookscan.app.engine

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tan

/**
 * 사진 전체를 보고 판단하는 단계 (engine/orient.py, engine/bookscan.py 의 find_gutter 와 같은 알고리즘·수치)
 *  - detectRotation: 글이 옆으로 누운 사진(폰을 세로로 들고 펼친 책을 찍은 경우)을 바로 세울 회전 각도
 *  - findGutter: 양면 펼침의 접힌 선 (두 페이지 글 사이 '글 없는 세로 띠' + 접힌 곳의 옅은 그림자)
 * 입력은 모두 RGB.
 */
object PageAnalysis {

    // ================================================================= 공통

    private class CC(val n: Int, val labels: IntArray?, val stats: IntArray, val cen: DoubleArray) {
        fun x(i: Int) = stats[i * 5 + Imgproc.CC_STAT_LEFT]
        fun y(i: Int) = stats[i * 5 + Imgproc.CC_STAT_TOP]
        fun w(i: Int) = stats[i * 5 + Imgproc.CC_STAT_WIDTH]
        fun h(i: Int) = stats[i * 5 + Imgproc.CC_STAT_HEIGHT]
        fun area(i: Int) = stats[i * 5 + Imgproc.CC_STAT_AREA]
        fun cx(i: Int) = cen[i * 2]
        fun cy(i: Int) = cen[i * 2 + 1]
    }

    private fun components(mask: Mat, wantLabels: Boolean): CC {
        val lbl = Mat()
        val st = Mat()
        val cen = Mat()
        val n = Imgproc.connectedComponentsWithStats(mask, lbl, st, cen, 8, CvType.CV_32S)
        val stats = IntArray(n * 5).also { st.get(0, 0, it) }
        val c = DoubleArray(n * 2).also { cen.get(0, 0, it) }
        val labels = if (wantLabels) IntArray(mask.rows() * mask.cols()).also { lbl.get(0, 0, it) } else null
        lbl.release(); st.release(); cen.release()
        return CC(n, labels, stats, c)
    }

    private fun bytes(m: Mat): ByteArray = ByteArray((m.total() * m.channels()).toInt()).also { m.get(0, 0, it) }

    private fun maskFrom(rows: Int, cols: Int, on: BooleanArray): Mat {
        val b = ByteArray(on.size) { if (on[it]) 255.toByte() else 0 }
        return Mat(rows, cols, CvType.CV_8UC1).also { it.put(0, 0, b) }
    }

    private fun rect(w: Int, h: Int): Mat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(w.toDouble(), h.toDouble()))

    private fun toGray(rgb: Mat, maxSide: Double, cap1: Boolean): Mat {
        val h = rgb.rows()
        val w = rgb.cols()
        val s = if (cap1) min(1.0, maxSide / max(h, w)) else maxSide / max(h, w)
        val gray = Mat()
        if (rgb.channels() == 3) Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY) else rgb.copyTo(gray)
        val g = Mat()
        Imgproc.resize(gray, g, Size(BookScanEngine.rint(w * s).toDouble(), BookScanEngine.rint(h * s).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        gray.release()
        return g
    }

    /** textlines.text_mask: 글자 픽셀=255 */
    private fun textMask(gray: Mat): Mat {
        val block = max(15, (min(gray.rows(), gray.cols()) * 0.03).toInt()) or 1
        val m = Mat()
        Imgproc.adaptiveThreshold(gray, m, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, block, 15.0)
        return m
    }

    /** textlines.char_height */
    private fun charHeight(cc: CC, rows: Int, cols: Int): Double? {
        val hs = ArrayList<Double>()
        for (i in 1 until cc.n) {
            val h = cc.h(i)
            if (h >= 4 && h < rows * 0.05 && cc.w(i) < cols * 0.1) hs.add(h.toDouble())
        }
        if (hs.size < 30) return null
        return BookScanEngine.median(hs.toDoubleArray())
    }

    // ================================================================= 방향 (orient.py)

    /** 글이 바로 서도록 시계 방향으로 돌릴 각도(0/90/180/270) */
    fun detectRotation(rgb: Mat): Int {
        val g = toGray(rgb, 1600.0, true)
        try {
            val m = textMask(g)
            val cc = components(m, true)
            val ch = charHeight(cc, m.rows(), m.cols())
            if (ch == null) {
                m.release(); return 0
            }
            val cm = charMask(cc, m.rows(), m.cols(), ch)
            m.release()
            val (sH, sV) = lineLikeness(cm, ch)
            cm.release()
            val cands = if (sV > sH * VERT_RATIO) intArrayOf(90, 270) else intArrayOf(0, 180)
            val (p0, _) = punctScoreRotated(g, cands[0])
            val (p1, n1) = punctScoreRotated(g, cands[1])
            if (cands[0] == 90) return if (p1 > p0) 270 else 90
            return if (n1 >= 8 && p1 > p0 + 0.3) 180 else 0
        } finally {
            g.release()
        }
    }

    /** RGB 사진을 시계 방향으로 deg 만큼 돌린 새 Mat */
    fun rotate(src: Mat, deg: Int): Mat {
        val out = Mat()
        when (deg) {
            90 -> Core.rotate(src, out, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(src, out, Core.ROTATE_180)
            270 -> Core.rotate(src, out, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> src.copyTo(out)
        }
        return out
    }

    private const val VERT_RATIO = 1.5

    /** 글자 크기·모양(가늘고 긴 조각 제외)인 덩어리만, 사진 가운데 80% 안 */
    private fun charMask(cc: CC, rows: Int, cols: Int, ch: Double): Mat {
        val ok = BooleanArray(cc.n) { i ->
            if (i == 0) return@BooleanArray false
            val big = max(cc.w(i), cc.h(i)).toDouble()
            val small = max(min(cc.w(i), cc.h(i)), 1).toDouble()
            big >= ch * 0.5 && big <= ch * 2.5 && big / small <= 4.0 && small >= ch * 0.25 && cc.area(i) >= 4 &&
                cc.cx(i) > cols * 0.1 && cc.cx(i) < cols * 0.9 && cc.cy(i) > rows * 0.1 && cc.cy(i) < rows * 0.9
        }
        val lab = cc.labels!!
        return maskFrom(rows, cols, BooleanArray(lab.size) { ok[lab[it]] })
    }

    /** 가로로 이어 붙인 글줄 총 길이(H), 세로로 이어 붙인 글줄 총 길이(V) */
    private fun lineLikeness(cm: Mat, ch: Double): Pair<Double, Double> {
        val k = max(3, (ch * 1.2).toInt())
        var hSum = 0.0
        var vSum = 0.0
        for (horiz in listOf(true, false)) {
            val ker = if (horiz) rect(k, 3) else rect(3, k) // 기울어진 글줄도 이어지게
            val b = Mat()
            Imgproc.morphologyEx(cm, b, Imgproc.MORPH_CLOSE, ker)
            ker.release()
            val cc = components(b, false)
            b.release()
            for (i in 1 until cc.n) {
                if (horiz) {
                    if (cc.w(i) >= ch * 5 && cc.h(i) <= ch * 2.2) hSum += cc.w(i)
                } else {
                    if (cc.h(i) >= ch * 5 && cc.w(i) <= ch * 2.2) vSum += cc.h(i)
                }
            }
        }
        return Pair(hSum, vSum)
    }

    private fun punctScoreRotated(g: Mat, deg: Int): Pair<Double, Int> {
        val r = rotate(g, deg)
        try {
            return punctScore(r)
        } finally {
            r.release()
        }
    }

    /** 글줄 띠 안에서 마침표·쉼표 같은 작은 점이 아래쪽에 있으면 +, 위쪽이면 −. (점수, 점 개수) */
    private fun punctScore(gray: Mat): Pair<Double, Int> {
        val m = textMask(gray)
        val h = m.rows()
        val w = m.cols()
        val ccM = components(m, true)
        val ch = charHeight(ccM, h, w)
        if (ch == null) {
            m.release(); return Pair(0.0, 0)
        }
        // clean_mask
        val keep = BooleanArray(ccM.n) { i ->
            i != 0 && ccM.h(i) <= ch * 2.5 && ccM.w(i) <= ch * 6 && ccM.area(i) >= max(4.0, ch * ch * 0.04)
        }
        val labM = ccM.labels!!
        val clean = maskFrom(h, w, BooleanArray(labM.size) { keep[labM[it]] })
        val kx = max(3, (ch * 1.5).toInt())
        val ker = rect(kx, 1)
        val blob = Mat()
        Imgproc.morphologyEx(clean, blob, Imgproc.MORPH_CLOSE, ker)
        ker.release(); clean.release()
        val ccL = components(blob, true)
        blob.release()
        val isLine = BooleanArray(ccL.n) { i ->
            i != 0 && ccL.w(i) >= ch * 4 && ccL.h(i) >= ch * 0.6 && ccL.h(i) <= ch * 1.8
        }
        val labL = ccL.labels!!
        val mb = bytes(m)
        m.release()

        fun anyOn(y0: Int, y1: Int, x0: Int, x1: Int): Boolean {
            for (y in y0 until y1) for (x in x0 until x1) if (mb[y * w + x].toInt() != 0) return true
            return false
        }

        var bottom = 0
        var top = 0
        val r = max(2, Math.rint(ch * 0.6).toInt())
        for (i in 1 until ccM.n) {
            val bw = ccM.w(i)
            val bh = ccM.h(i)
            val ar = ccM.area(i)
            if (!(ar >= 2 && ar <= ch * ch * 0.12 && max(bw, bh) <= ch * 0.4)) continue
            val cx = Math.rint(ccM.cx(i)).toInt()
            val cy = Math.rint(ccM.cy(i)).toInt()
            // 바로 위·아래에 글자 획이 있는 점(i, j 의 점 등)은 문장부호가 아님
            val x0 = ccM.x(i)
            val x1 = min(w, x0 + bw)
            val ya0 = max(0, (cy - ch * 0.9).toInt())
            val ya1 = max(0, (cy - ch * 0.3).toInt())
            val yb0 = min(h, (cy + ch * 0.3).toInt())
            val yb1 = min(h, (cy + ch * 0.9).toInt())
            if (anyOn(ya0, ya1, x0, x1) || anyOn(yb0, yb1, x0, x1)) continue
            if (cy < 0 || cy >= h) continue
            var best = -1
            for (dx in intArrayOf(-r, r, -2 * r, 2 * r)) {
                val xx = cx + dx
                if (xx in 0 until w) {
                    val j = labL[cy * w + xx]
                    if (j > 0 && isLine[j]) {
                        best = j; break
                    }
                }
            }
            if (best < 0) continue
            val rel = (cy - ccL.y(best)).toDouble() / max(ccL.h(best), 1)
            if (rel > 0.62) bottom++ else if (rel < 0.38) top++
        }
        val tot = bottom + top
        return Pair((bottom - top).toDouble() / max(1, tot), tot)
    }

    // ================================================================= 양면 접힌 선 (bookscan.find_gutter)

    /** 양면 펼침이면 접힌 선 x = a*y + b (원본 좌표), 아니면 null */
    fun findGutter(rgb: Mat, force: Boolean): Pair<Double, Double>? {
        val h = rgb.rows()
        val w = rgb.cols()
        val s = 800.0 / max(h, w)
        val gray = toGray(rgb, 800.0, false)
        val sh = gray.rows()
        val sw = gray.cols()
        try {
            if (!force && sw < sh * 1.05) return null // 세로로 긴 사진은 한 페이지
            val txt = textMaskSmall(gray)
            val cx = (sw - 1) / 2.0
            val nb = 14
            val ys = IntArray(nb + 1) { (sh * 0.06 + it * (sh * 0.88) / nb).toInt() }
            val yc = DoubleArray(nb) { (ys[it] + ys[it + 1]) / 2.0 }
            val cap = max(3.0, sw * 0.03)
            val lo = max(0.0, cx - sw * 0.22).toInt()
            val hi = min(sw - 1.0, cx + sw * 0.22).toInt()
            val clear = Array(nb) { DoubleArray(sw) }
            var informative = 0
            for (k in 0 until nb) {
                val dens = IntArray(sw)
                for (y in ys[k] until ys[k + 1]) for (x in 0 until sw) if (txt[y * sw + x]) dens[x]++
                val occ = BooleanArray(sw) { x ->
                    (if (x > 0) dens[x - 1] else 0) + dens[x] + (if (x < sw - 1) dens[x + 1] else 0) > 0
                }
                clearance1d(occ, cap, clear[k])
                var anyClear = false
                var anyOcc = false
                for (x in lo..hi) {
                    if (clear[k][x] > 0) anyClear = true
                    if (occ[x]) anyOcc = true
                }
                if (anyClear && anyOcc) informative++ // 글과 빈 칸이 다 있는 띠만 판단에 씀
            }
            val span = yc[nb - 1] - yc[0]
            val maxDx = tan(Math.toRadians(15.0)) * span
            val tk = DoubleArray(nb) { (yc[it] - yc[0]) / span }
            var bestScore = -Double.MAX_VALUE
            var bestT = lo
            var bestB = lo
            var bestHits = 0
            for (xt in lo..hi) {
                for (xb in lo..hi) {
                    if (abs(xb - xt) > maxDx) continue
                    var sc = 0.0
                    var hits = 0
                    for (k in 0 until nb) {
                        val xk = Math.rint(xt + (xb - xt) * tk[k]).toInt().coerceIn(0, sw - 1)
                        val c = clear[k][xk]
                        sc += c
                        if (c > 0) hits++
                    }
                    sc -= 0.5 * cap * abs((xt + xb) / 2.0 - cx) / sw * nb // 같은 점수면 가운데 쪽
                    if (sc > bestScore) {
                        bestScore = sc; bestT = xt; bestB = xb; bestHits = hits
                    }
                }
            }
            val a = (bestB - bestT) / span
            val b = bestT - a * yc[0]
            val xm = (bestT + bestB) / 2.0
            if (!force) {
                val frac = bestHits.toDouble() / max(1, informative)
                var left = 0L
                var right = 0L
                val split = max(1, xm.toInt())
                for (y in 0 until sh) for (x in 0 until sw) if (txt[y * sw + x]) {
                    if (x < split) left++
                    if (x >= xm.toInt()) right++
                }
                val balance = min(left, right).toDouble() / max(1L, max(left, right))
                if (informative < 6 || frac < 0.7 || abs(xm - cx) > sw * 0.15 || balance < 0.25) return null
                // 진짜 접힌 선에는 종이가 꺾이며 생기는 옅은 그림자 골짜기가 있다(표의 칸 사이 빈 줄과 구분)
                if (foldShadow(gray, a, b) < 0.012) return null
            }
            return Pair(a, b / s)
        } finally {
            gray.release()
        }
    }

    /** bookscan._text_mask_small: 글자 픽셀(true). 그림·상자 테두리 같은 큰 덩어리는 제외 */
    private fun textMaskSmall(gray: Mat): BooleanArray {
        val sh = gray.rows()
        val sw = gray.cols()
        val block = max(15, (min(sh, sw) * 0.03).toInt()) or 1
        val m = Mat()
        Imgproc.adaptiveThreshold(gray, m, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, block, 12.0)
        val cc = components(m, true)
        m.release()
        val keep = BooleanArray(cc.n) { i -> i != 0 && cc.h(i) < sh * 0.04 && cc.w(i) < sw * 0.04 && cc.area(i) >= 3 }
        val lab = cc.labels!!
        return BooleanArray(lab.size) { keep[lab[it]] }
    }

    /** 빈 열에서 가장 가까운 글자 열까지 거리(최대 cap). 글자 열은 0 */
    private fun clearance1d(occ: BooleanArray, cap: Double, out: DoubleArray) {
        val n = occ.size
        var last = -1e9
        for (x in 0 until n) {
            if (occ[x]) last = x.toDouble()
            out[x] = x - last
        }
        last = 1e9
        for (x in n - 1 downTo 0) {
            if (occ[x]) last = x.toDouble()
            out[x] = min(out[x], last - x)
        }
        for (x in 0 until n) out[x] = min(out[x], cap)
    }

    /** 직선 x = a*y + b (작은 이미지 좌표) 를 따라 잰 그림자 골짜기 깊이(중앙값, 종이 밝기 대비 비율) */
    private fun foldShadow(gray: Mat, a: Double, b: Double): Double {
        val sh = gray.rows()
        val sw = gray.cols()
        val k = max(7, (sw * 0.025).toInt()) or 1
        val ker = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))
        val paperM = Mat()
        Imgproc.morphologyEx(gray, paperM, Imgproc.MORPH_CLOSE, ker)
        ker.release()
        val paper = bytes(paperM)
        paperM.release()
        val wv = max(6, (sw * 0.05).toInt())
        val rels = ArrayList<Double>()
        for (i in 0 until 12) {
            val yy = sh * 0.1 + i * (sh * 0.8) / 11.0
            val y0 = yy.toInt()
            val r0 = max(0, y0 - 8)
            val r1 = min(sh, y0 + 8)
            val xl = Math.rint(a * yy + b).toInt()
            if (xl - wv < 0 || xl + wv >= sw) continue
            fun band(x: Int): Double {
                val v = DoubleArray(r1 - r0) { (paper[(r0 + it) * sw + x].toInt() and 0xFF).toDouble() }
                return BookScanEngine.median(v)
            }
            var c = Double.MAX_VALUE
            for (x in xl - 3..xl + 3) c = min(c, band(x))
            var lm = -Double.MAX_VALUE
            for (x in xl - wv until xl - 3) lm = max(lm, band(x))
            var rm = -Double.MAX_VALUE
            for (x in xl + 4 until xl + wv) rm = max(rm, band(x))
            val side = min(lm, rm)
            rels.add((side - c) / max(side, 1.0))
        }
        return if (rels.isEmpty()) 0.0 else BookScanEngine.median(rels.toDoubleArray())
    }
}
