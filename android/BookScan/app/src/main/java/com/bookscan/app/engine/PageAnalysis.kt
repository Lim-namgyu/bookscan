package com.bookscan.app.engine

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tan

/**
 * 사진 전체를 보고 판단하는 단계 (engine/orient.py, engine/bookscan.py 의 find_gutter 와 같은 알고리즘·수치)
 *  - detectRotation: 글이 옆으로 눕거나 거꾸로 된 사진을 바로 세울 회전 각도
 *  - findGutter: 양면 펼침의 접힌 선 (책 윤곽의 가운데 선 근처에서, 두 페이지 글 사이 '글 없는 세로 띠' + 접힌 곳의 옅은 그림자)
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

    /**
     * 글이 바로 서도록 시계 방향으로 돌릴 각도(0/90/180/270).
     * 가로/세로 글 판단 → 글줄이 수평이 되도록 기울기를 바로잡은 뒤(기울어진 채로는 점의 위아래 판정이 틀어짐)
     * 두 단서의 합: 문장부호가 글줄 아래쪽에 있는가 + 줄 시작(왼쪽)이 가지런한가.
     */
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
            val vert = sV > sH * VERT_RATIO
            val base = if (vert) rotate(g, 90) else g
            val d = deskew(base)
            if (base !== g) base.release()
            try {
                val (p0, n0) = punctScore(d)
                val (p1, n1) = punctScoreRotated(d, 180)
                var vote = alignScore(d) * 2.0
                if (n0 + n1 >= 6) vote += (p0 - p1) / 2.0
                if (vert) return if (vote < 0) 270 else 90
                return if (vote < -FLIP_MIN) 180 else 0
            } finally {
                d.release()
            }
        } finally {
            g.release()
        }
    }

    /** orient.deskew: 글줄이 수평이 되도록 돌린 새 Mat */
    private fun deskew(g: Mat): Mat {
        val m = textMask(g)
        val cc = components(m, true)
        val ch = charHeight(cc, m.rows(), m.cols())
        if (ch == null) {
            m.release(); return g.clone()
        }
        val cm = charMask(cc, m.rows(), m.cols(), ch)
        m.release()
        val sk = Straightener.skewOfMask(cm, ch, 20.0)
        cm.release()
        val w = g.cols()
        val h = g.rows()
        val rm = Imgproc.getRotationMatrix2D(Point(w / 2.0, h / 2.0), sk, 1.0)
        val out = Mat()
        Imgproc.warpAffine(g, out, rm, Size(w.toDouble(), h.toDouble()), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
        rm.release()
        return out
    }

    /**
     * orient.align_score: 줄 시작이 줄 끝보다 잘 맞춰져 있으면 +, 반대면 − (−1~1).
     * 가로쓰기는 왼쪽 정렬이고 번호·문단 끝 짧은 줄 때문에 오른쪽 끝은 들쭉날쭉하다.
     */
    private fun alignScore(g: Mat): Double {
        val (lines, ch) = Straightener.lineEnds(g, 0.04)
        if (ch == null || lines.size < 6) return 0.0
        val tol = ch * 0.6
        var s = 0L
        var e = 0L
        for (i in lines.indices) for (j in lines.indices) {
            if (i == j || abs(lines[j][2] - lines[i][2]) >= ch * 12) continue // 같은 단·가까운 줄끼리만
            if (abs(lines[j][0] - lines[i][0]) < tol) s++
            if (abs(lines[j][1] - lines[i][1]) < tol) e++
        }
        return (s - e).toDouble() / max(1L, s + e)
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

    /** 가로 글을 거꾸로(180°) 돌리려면 이만큼 뚜렷한 근거가 있어야 함 (거꾸로 찍는 일은 드묾) */
    private const val FLIP_MIN = 0.5

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

    /** numpy.linspace(a, b, n)[i] 와 비트 단위로 같은 값 */
    private fun lin(a: Double, b: Double, n: Int, i: Int): Double = if (i == n - 1) b else i * ((b - a) / (n - 1)) + a

    /**
     * 양면 펼침이면 접힌 선 x = a*y + b (원본 좌표), 아니면 null.
     * 접힌 선 = 두 페이지 글 사이의 '글 없는 세로 띠'를 위에서 아래까지 가장 여유 있게 지나는 직선.
     * mask(책 영역, rgb 와 같은 크기)가 있으면: 책이 가로로 긴지(양면)로 판단하고, 책 바깥 변의 가운데 선 근처에서만 찾는다.
     */
    fun findGutter(rgb: Mat, force: Boolean, mask: Mat? = null): Pair<Double, Double>? {
        val h = rgb.rows()
        val w = rgb.cols()
        val s = 800.0 / max(h, w)
        val small = Mat()
        Imgproc.resize(rgb, small, Size(0.0, 0.0), s, s, Imgproc.INTER_AREA)
        val gray = Mat()
        if (small.channels() == 3) Imgproc.cvtColor(small, gray, Imgproc.COLOR_RGB2GRAY) else small.copyTo(gray)
        small.release()
        val sh = gray.rows()
        val sw = gray.cols()
        try {
            var axis: DoubleArray? = null
            if (mask != null) {
                val ms = Mat()
                Imgproc.resize(mask, ms, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
                axis = bookAxis(ms)
                ms.release()
            }
            if (axis != null) {
                if (!force && axis[2] < SPREAD_ASPECT) return null // 책이 세로로 길면 한 쪽
            } else if (!force && sw < sh * 1.05) {
                return null // 세로로 긴 사진은 한 페이지
            }
            val txt = textMaskSmall(gray)
            val cx = (sw - 1) / 2.0
            val nb = 14
            val ys = IntArray(nb + 1) { lin(sh * 0.06, sh * 0.94, nb + 1, it).toInt() }
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
            val nx = hi - lo + 1
            val score = DoubleArray(nx * nx)
            val hits = IntArray(nx * nx)
            val pt = if (axis != null) axis[0] * yc[0] + axis[1] else 0.0
            val pb = if (axis != null) axis[0] * yc[nb - 1] + axis[1] else 0.0
            val cxx = if (axis != null) (pt + pb) / 2.0 else cx
            var best = -Double.MAX_VALUE
            for (i in 0 until nx) {
                val xt = lo + i
                for (j in 0 until nx) {
                    val xb = lo + j
                    val idx = i * nx + j
                    var sc = 0.0
                    var ht = 0
                    for (k in 0 until nb) {
                        val xk = Math.rint(xt + (xb - xt) * tk[k]).toInt().coerceIn(0, sw - 1)
                        val c = clear[k][xk]
                        sc += c
                        if (c > 0) ht++
                    }
                    hits[idx] = ht
                    if (axis != null) {
                        // 책 가운데 선 근처만, 가까울수록 우선
                        val dev = (abs(xt - pt) + abs(xb - pb)) / 2.0
                        sc -= AXIS_PULL * cap * dev / sw * nb
                        if (abs(xt - pt) > sw * AXIS_WIN || abs(xb - pb) > sw * AXIS_WIN) sc = -1e18
                    } else {
                        sc -= 0.5 * cap * abs((xt + xb) / 2.0 - cx) / sw * nb // 같은 점수면 가운데 쪽
                    }
                    if (abs(xb - xt) > maxDx) sc = -1e18
                    score[idx] = sc
                    if (sc > best) best = sc
                }
            }
            if (best < -1e17) return null
            // 빈 띠 점수가 최고점 근처인 후보들을 점수 순으로 보며, '접힌 그림자'가 있는 첫 직선을 고른다.
            val tol = GUTTER_TOL * nb * cap
            val near = (0 until nx * nx).filter { score[it] >= best - tol }
                .sortedWith(compareByDescending<Int> { score[it] }.thenByDescending { it })
            val cands = ArrayList<Int>()
            for (idx in near) {
                val i = idx / nx
                val j = idx % nx
                if (cands.any { abs(i - it / nx) <= 2 && abs(j - it % nx) <= 2 }) continue
                cands.add(idx)
                if (cands.size >= GUTTER_CANDS) break
            }
            val paper = foldPaper(gray)
            var kSel = 0
            var fold = 0.0
            for ((n, idx) in cands.withIndex()) {
                val xt = (lo + idx / nx).toDouble()
                val xb = (lo + idx % nx).toDouble()
                val a0 = (xb - xt) / span
                val f = foldShadow(paper, sh, sw, a0, xt - a0 * yc[0])
                if (n == 0) fold = f
                if (f >= FOLD_MIN) {
                    kSel = n; fold = f; break
                }
            }
            val sel = cands[kSel]
            val bestT = (lo + sel / nx).toDouble()
            val bestB = (lo + sel % nx).toDouble()
            val a = (bestB - bestT) / span
            val b = bestT - a * yc[0]
            val xm = (bestT + bestB) / 2.0
            if (!force) {
                val frac = hits[sel].toDouble() / max(1, informative)
                var left = 0L
                var right = 0L
                val split = max(1, xm.toInt())
                for (y in 0 until sh) for (x in 0 until sw) if (txt[y * sw + x]) {
                    if (x < split) left++
                    if (x >= xm.toInt()) right++
                }
                val balance = min(left, right).toDouble() / max(1L, max(left, right))
                if (informative < 6 || frac < 0.7 || abs(xm - cxx) > sw * 0.15 || balance < 0.25) return null
                // 진짜 접힌 선에는 종이가 꺾이며 생기는 옅은 그림자 골짜기가 있다(표의 칸 사이 빈 줄과 구분)
                if (fold < FOLD_MIN) return null
            }
            return Pair(a, b / s)
        } finally {
            gray.release()
        }
    }

    private const val SPREAD_ASPECT = 1.1 // 책(마스크)이 이보다 가로로 길어야 양면 후보
    private const val AXIS_PULL = 2.0     // 책 가운데 선에서 벗어난 만큼 깎는 점수(빈 띠 점수 단위)
    private const val AXIS_WIN = 0.07     // 책 가운데 선에서 이 폭(사진 폭 비율) 안에서만 찾음
    private const val GUTTER_TOL = 0.2    // 최고점에서 (띠 수 x 최대 여유)의 이 비율 안쪽이면 후보
    private const val GUTTER_CANDS = 24
    private const val FOLD_MIN = 0.012

    /** x = a*y + b, 튀는 점은 빼고 두 번 다시 맞춤 */
    private fun robustLine(Y: DoubleArray, X: DoubleArray): Pair<Double, Double>? {
        var keep = BooleanArray(Y.size) { true }
        fun fit(): Pair<Double, Double>? {
            val ys = ArrayList<Double>()
            val xs = ArrayList<Double>()
            for (i in Y.indices) if (keep[i]) {
                ys.add(Y[i]); xs.add(X[i])
            }
            return BookScanEngine.polyfit1(ys, xs)
        }
        var ab = fit() ?: return null
        repeat(2) {
            val r = DoubleArray(Y.size) { X[it] - (ab.first * Y[it] + ab.second) }
            val med = BookScanEngine.median(r)
            val mad = BookScanEngine.median(DoubleArray(r.size) { abs(r[it] - med) }) + 1e-9
            val k2 = BooleanArray(Y.size) { abs(r[it]) < max(2.0, 3.0 * mad) }
            if (k2.count { it } < 5) return ab
            keep = k2
            ab = fit() ?: return ab
        }
        return ab
    }

    /**
     * bookscan.book_axis: 책 마스크(작은 크기)에서 양면의 가운데 선 (a, b: x = a*y + b) 과 가로세로비.
     * 가운데 높이(30~70%) 행마다 책의 왼쪽·오른쪽 바깥 변의 중점 → 두 페이지 폭이 같으므로 접힌 선은 거의 이 선 위.
     * 책이 사진 밖으로 잘려 믿을 수 없으면 null
     */
    private fun bookAxis(maskSmall: Mat): DoubleArray? {
        val sh = maskSmall.rows()
        val sw = maskSmall.cols()
        val mb = bytes(maskSmall)
        fun on(y: Int, x: Int) = (mb[y * sw + x].toInt() and 0xFF) > 127
        var y0 = -1
        var y1 = -1
        var nRows = 0
        for (y in 0 until sh) {
            var any = false
            for (x in 0 until sw) if (on(y, x)) {
                any = true; break
            }
            if (any) {
                if (y0 < 0) y0 = y
                y1 = y; nRows++
            }
        }
        if (nRows < 20) return null
        val Y = ArrayList<Double>()
        val L = ArrayList<Double>()
        val R = ArrayList<Double>()
        for (y in (y0 + 0.3 * (y1 - y0)).toInt()..(y0 + 0.7 * (y1 - y0)).toInt()) {
            var l = -1
            var r = -1
            for (x in 0 until sw) if (on(y, x)) {
                if (l < 0) l = x
                r = x
            }
            if (l >= 0) {
                Y.add(y.toDouble()); L.add(l.toDouble()); R.add(r.toDouble())
            }
        }
        if (Y.size < 10) return null
        if (L.count { it <= 1 } > 0.3 * L.size || R.count { it >= sw - 2 } > 0.3 * R.size) return null // 책 바깥 변이 사진 밖
        val yArr = Y.toDoubleArray()
        val (al, bl) = robustLine(yArr, L.toDoubleArray()) ?: return null
        val (ar, br) = robustLine(yArr, R.toDoubleArray()) ?: return null
        // 가로세로비: 책을 감싸는 가장 작은 회전 사각형의 (가로에 가까운 변) / (세로에 가까운 변)
        val bin = Mat()
        Imgproc.threshold(maskSmall, bin, 127.0, 255.0, Imgproc.THRESH_BINARY)
        val cnts = ArrayList<MatOfPoint>()
        val hier = Mat()
        Imgproc.findContours(bin, cnts, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hier.release(); bin.release()
        if (cnts.isEmpty()) return null
        val c = cnts.maxByOrNull { Imgproc.contourArea(it) }!!
        val c2 = MatOfPoint2f(*c.toArray())
        val rr = Imgproc.minAreaRect(c2)
        c2.release()
        cnts.forEach { it.release() }
        val p = arrayOfNulls<Point>(4)
        rr.points(p)
        val v1x = p[1]!!.x - p[0]!!.x
        val v1y = p[1]!!.y - p[0]!!.y
        val v2x = p[2]!!.x - p[1]!!.x
        val v2y = p[2]!!.y - p[1]!!.y
        val n1 = hypot(v1x, v1y)
        val n2 = hypot(v2x, v2y)
        if (n1 < 1 || n2 < 1) return null
        val aspect = if (abs(v1x) / n1 >= abs(v2x) / n2) n1 / n2 else n2 / n1
        return doubleArrayOf((al + ar) / 2.0, (bl + br) / 2.0, aspect)
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

    /** 글자·가는 선을 지운 종이 밝기 (8비트, 행 우선) */
    private fun foldPaper(gray: Mat): ByteArray {
        val sw = gray.cols()
        val k = max(7, (sw * 0.025).toInt()) or 1
        val ker = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))
        val paperM = Mat()
        Imgproc.morphologyEx(gray, paperM, Imgproc.MORPH_CLOSE, ker)
        ker.release()
        val paper = bytes(paperM)
        paperM.release()
        return paper
    }

    /** 직선 x = a*y + b (작은 이미지 좌표) 를 따라 잰 그림자 골짜기 깊이(중앙값, 종이 밝기 대비 비율) */
    private fun foldShadow(paper: ByteArray, sh: Int, sw: Int, a: Double, b: Double): Double {
        val wv = max(6, (sw * 0.05).toInt())
        val rels = ArrayList<Double>()
        for (i in 0 until 12) {
            val yy = lin(sh * 0.1, sh * 0.9, 12, i)
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
