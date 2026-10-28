package com.bookscan.app.engine

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * 글줄 기반 반듯하게 펴기 (곡면 펴기 모델이 남긴 기울기·휨·좌우 쏠림 제거).
 * engine/straighten.py + engine/textlines.py 와 같은 알고리즘·같은 수치.
 *
 *  1) 글줄 기울기·휨 → 세로 이동 지도 (글자 세로획 방향은 건드리지 않음)
 *  2) 글자 세로획 기울기 → 가로 이동 (좌우 여백선이 있으면 여백선 기준, 없으면 세로획 각도)
 *  페이지를 통째로 회전하지 않는다. 글줄만 기울고 글자는 똑바로 선 경우 회전하면 글자가 이탤릭처럼 눕기 때문.
 *
 * 결과는 '출력 좌표 → 입력 좌표' 지도(CV_32F 두 장). 보정으로 밀려나는 내용(머리글·쪽번호)이
 * 잘리지 않도록 입력보다 조금 클 수 있다.
 */
object Straightener {

    class Maps(val mapX: Mat, val mapY: Mat)

    /** 입력: RGB(8UC3) 또는 GRAY(8UC1). 글자가 너무 적으면 null */
    fun maps(img: Mat): Maps? {
        val gray = Mat()
        if (img.channels() == 3) Imgproc.cvtColor(img, gray, Imgproc.COLOR_RGB2GRAY) else img.copyTo(gray)
        var clean: Mat? = null
        try {
            val mask = textMask(gray)
            val cc = components(mask, true)
            val ch = charHeight(cc, mask.rows(), mask.cols())
            if (ch == null) {
                mask.release(); return null
            }
            clean = cleanMask(cc, mask.rows(), mask.cols(), ch)
            mask.release()
            val cen = Centroids.of(components(clean, false))
            val (aH, sH) = skewSearch(cen, ch, 0.0, 30.0)
            val (aV, sV) = skewSearch(cen, ch, 90.0, 30.0)
            if (sV > sH * 1.15) {
                // 옆으로 누운 글(가로로 인쇄된 표 등): 페이지는 그대로 두고, 전치한 상태에서 같은 처리
                val gT = Mat(); Core.transpose(gray, gT)
                val cT = Mat(); Core.transpose(clean, cT)
                val r = try {
                    core(gT, cT, ch, 90.0 - aV)
                } finally {
                    gT.release(); cT.release()
                }
                val mx = Mat(); Core.transpose(r.mapY, mx)
                val my = Mat(); Core.transpose(r.mapX, my)
                r.mapX.release(); r.mapY.release()
                return Maps(mx, my)
            }
            return core(gray, clean, ch, aH)
        } finally {
            gray.release(); clean?.release()
        }
    }

    // ================================================================= 글자·글줄 찾기 (textlines.py)

    private fun textMask(gray: Mat): Mat {
        val block = max(15, (min(gray.rows(), gray.cols()) * 0.03).toInt()) or 1
        val m = Mat()
        Imgproc.adaptiveThreshold(gray, m, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, block, 15.0)
        return m
    }

    private class CC(val n: Int, val labels: IntArray?, val stats: IntArray, val cx: DoubleArray, val cy: DoubleArray) {
        fun x(i: Int) = stats[i * 5 + Imgproc.CC_STAT_LEFT]
        fun y(i: Int) = stats[i * 5 + Imgproc.CC_STAT_TOP]
        fun w(i: Int) = stats[i * 5 + Imgproc.CC_STAT_WIDTH]
        fun h(i: Int) = stats[i * 5 + Imgproc.CC_STAT_HEIGHT]
        fun area(i: Int) = stats[i * 5 + Imgproc.CC_STAT_AREA]
    }

    private fun components(mask: Mat, wantLabels: Boolean): CC {
        val lbl = Mat()
        val st = Mat()
        val cen = Mat()
        val n = Imgproc.connectedComponentsWithStats(mask, lbl, st, cen, 8, CvType.CV_32S)
        val stats = IntArray(n * 5)
        st.get(0, 0, stats)
        val c = DoubleArray(n * 2)
        cen.get(0, 0, c)
        val labels = if (wantLabels) IntArray(mask.rows() * mask.cols()).also { lbl.get(0, 0, it) } else null
        lbl.release(); st.release(); cen.release()
        return CC(n, labels, stats, DoubleArray(n) { c[it * 2] }, DoubleArray(n) { c[it * 2 + 1] })
    }

    private fun charHeight(cc: CC, rows: Int, cols: Int): Double? {
        val hs = ArrayList<Double>()
        for (i in 1 until cc.n) {
            val h = cc.h(i)
            if (h >= 4 && h < rows * 0.05 && cc.w(i) < cols * 0.1) hs.add(h.toDouble())
        }
        if (hs.size < 30) return null
        return median(hs.toDoubleArray())
    }

    /** 글자 크기와 동떨어진 덩어리(그림, 테두리, 얼룩, 점 잡음) 제거 */
    private fun cleanMask(cc: CC, rows: Int, cols: Int, ch: Double): Mat {
        val keep = BooleanArray(cc.n) { i ->
            i != 0 && cc.h(i) <= ch * 2.5 && cc.w(i) <= ch * 6 && cc.area(i) >= max(4.0, ch * ch * 0.04)
        }
        val lab = cc.labels!!
        val out = ByteArray(lab.size) { if (keep[lab[it]]) 255.toByte() else 0 }
        val m = Mat(rows, cols, CvType.CV_8UC1)
        m.put(0, 0, out)
        return m
    }

    private fun rect(w: Int, h: Int): Mat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(w.toDouble(), h.toDouble()))

    private class Samples(val xs: DoubleArray, val ys: DoubleArray)

    /** 글줄 중심선 표본. 반환: (글줄들, 글자 높이) */
    private fun detectLines(gray: Mat, minFrac: Double): Pair<List<Samples>, Double?> {
        val m0 = textMask(gray)
        val cc0 = components(m0, true)
        val ch = charHeight(cc0, m0.rows(), m0.cols())
        if (ch == null) {
            m0.release(); return Pair(emptyList(), null)
        }
        val mask = cleanMask(cc0, m0.rows(), m0.cols(), ch)
        m0.release()
        val h = mask.rows()
        val w = mask.cols()
        val kx = max(3, (ch * 2.2).toInt())
        val blob = Mat()
        val k1 = rect(kx, 1)
        val k2 = rect(max(3, ch.toInt()), 1)
        Imgproc.morphologyEx(mask, blob, Imgproc.MORPH_CLOSE, k1)
        Imgproc.morphologyEx(blob, blob, Imgproc.MORPH_OPEN, k2)
        k1.release(); k2.release()
        val cc = components(blob, true)
        blob.release()
        val mb = ByteArray(h * w)
        mask.get(0, 0, mb)
        mask.release()
        val lab = cc.labels!!
        val step = max(2, ch.toInt())
        val out = ArrayList<Samples>()

        for (i in 1 until cc.n) {
            val x = cc.x(i)
            val y = cc.y(i)
            val bw = cc.w(i)
            val bh = cc.h(i)
            if (bw < w * minFrac || bw < ch * 8) continue
            if (bh > ch * 2.6 + bw * 0.06) continue // 여러 줄이 엉긴 덩어리
            fun sub(r: Int, c: Int): Boolean {
                val idx = (y + r) * w + x + c
                return lab[idx] == i && mb[idx].toInt() != 0
            }
            var colsAny = 0
            for (c in 0 until bw) {
                for (r in 0 until bh) if (sub(r, c)) {
                    colsAny++; break
                }
            }
            if (colsAny < bw * 0.5) continue

            fun centers(band: Poly?): Samples {
                val xs = ArrayList<Double>()
                val ys = ArrayList<Double>()
                var c0 = 0
                while (c0 < bw) {
                    val c1 = min(bw, c0 + step)
                    val xc = c0 + (c1 - c0) / 2.0
                    val yc = band?.eval(xc)
                    var cnt = 0
                    var sumY = 0.0
                    for (r in 0 until bh) {
                        // 글줄 띠 안의 픽셀만: g, p, y 꼬리나 쉼표가 중심을 끌어내리지 않게
                        if (yc != null && abs(r - yc) >= ch * 0.55) continue
                        for (c in c0 until c1) if (sub(r, c)) {
                            cnt++; sumY += r
                        }
                    }
                    if (cnt >= ch * 0.5) {
                        xs.add(x + xc); ys.add(y + sumY / cnt)
                    }
                    c0 += step
                }
                return Samples(xs.toDoubleArray(), ys.toDoubleArray())
            }

            var s = centers(null)
            if (s.xs.size < 6) continue
            val deg = min(3, max(1, s.xs.size / 8))
            val band = polyfit(DoubleArray(s.xs.size) { s.xs[it] - x }, DoubleArray(s.ys.size) { s.ys[it] - y }, deg) ?: continue
            s = centers(band)
            if (s.xs.size < 6) continue
            // 이웃 중앙값으로 다듬기 (끝부분은 창을 줄여서)
            val n = s.ys.size
            val sm = DoubleArray(n) { j ->
                val r = min(2, min(j, n - 1 - j))
                median(s.ys.copyOfRange(j - r, j + r + 1))
            }
            out.add(Samples(s.xs, sm))
        }
        return Pair(out, ch)
    }

    /** 가로 괘선(밑줄·표 테두리·꼬리말 선)의 중심선 */
    private fun detectRules(gray: Mat, ch: Double, minFrac: Double = 0.25): List<Samples> {
        val mask = textMask(gray)
        val w = mask.cols()
        val k = max(9, (ch * 4).toInt()) or 1
        val horiz = Mat()
        val kr = rect(k, 1)
        Imgproc.morphologyEx(mask, horiz, Imgproc.MORPH_OPEN, kr)
        kr.release()
        mask.release()
        val cc = components(horiz, true)
        horiz.release()
        val lab = cc.labels!!
        val step = max(2, ch.toInt())
        val out = ArrayList<Samples>()
        for (i in 1 until cc.n) {
            val x = cc.x(i)
            val y = cc.y(i)
            val bw = cc.w(i)
            val bh = cc.h(i)
            if (bw < w * minFrac) continue
            val thick = cc.area(i).toDouble() / max(bw, 1)
            if (thick > ch * 0.9 || bh > bw * 0.2) continue
            val xs = ArrayList<Double>()
            val ys = ArrayList<Double>()
            var c0 = 0
            while (c0 < bw) {
                val c1 = min(bw, c0 + step)
                var cnt = 0
                var sumY = 0.0
                for (r in 0 until bh) for (c in c0 until c1) {
                    if (lab[(y + r) * w + x + c] == i) {
                        cnt++; sumY += r
                    }
                }
                if (cnt >= 1) {
                    xs.add(x + c0 + (c1 - c0) / 2.0); ys.add(y + sumY / cnt)
                }
                c0 += step
            }
            if (xs.size >= 6) out.add(Samples(xs.toDoubleArray(), ys.toDoubleArray()))
        }
        return out
    }

    // ================================================================= 기울기 찾기

    /** center±maxDeg 범위에서 글줄이 가장 또렷하게 겹치는 각도와 그 점수 */
    private class Centroids(val xs: DoubleArray, val ys: DoubleArray, val wt: DoubleArray) {
        companion object {
            fun of(cc: CC): Centroids {
                val n = cc.n - 1
                return Centroids(DoubleArray(n) { cc.cx[it + 1] }, DoubleArray(n) { cc.cy[it + 1] }, DoubleArray(n) { cc.area(it + 1).toDouble() })
            }
        }
    }

    private fun skewSearch(cen: Centroids, ch: Double, center: Double, maxDeg: Double): Pair<Double, Double> {
        val n = cen.xs.size
        if (n < 30) return Pair(center, 0.0)
        val xs = cen.xs
        val ys = cen.ys
        val wt = cen.wt
        val binW = max(1.0, ch * 0.5)
        val p = DoubleArray(n)

        fun score(deg: Double): Double {
            val t = Math.toRadians(deg)
            val s = sin(t)
            val c = cos(t)
            var pmin = Double.MAX_VALUE
            var pmax = -Double.MAX_VALUE
            for (k in 0 until n) {
                p[k] = -xs[k] * s + ys[k] * c
                if (p[k] < pmin) pmin = p[k]
                if (p[k] > pmax) pmax = p[k]
            }
            val hist = DoubleArray(floor((pmax - pmin) / binW).toInt() + 1)
            for (k in 0 until n) hist[floor((p[k] - pmin) / binW).toInt()] += wt[k]
            var sum = 0.0
            for (v in hist) sum += v * v
            return sum
        }

        val nCoarse = (2 * maxDeg / 0.5).roundToInt() + 1
        var best = center - maxDeg
        var bestS = -1.0
        for (k in 0 until nCoarse) {
            val a = center - maxDeg + 0.5 * k
            val sc = score(a)
            if (sc > bestS) {
                bestS = sc; best = a
            }
        }
        var fineBest = best - 0.6
        var fineS = -1.0
        for (k in 0..24) {
            val a = best - 0.6 + 0.05 * k
            val sc = score(a)
            if (sc > fineS) {
                fineS = sc; fineBest = a
            }
        }
        return Pair(fineBest, fineS)
    }

    /** 글자 세로 획의 기울기(도). 양수 = 획 위쪽이 왼쪽으로 누움 */
    private fun stemAngle(gray: Mat, clean: Mat): Double {
        val gxM = Mat()
        val gyM = Mat()
        Imgproc.Sobel(gray, gxM, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(gray, gyM, CvType.CV_32F, 0, 1, 3)
        val nearM = Mat()
        val k3 = Mat.ones(3, 3, CvType.CV_8UC1)
        Imgproc.dilate(clean, nearM, k3)
        k3.release()
        val total = gray.rows() * gray.cols()
        val gx = FloatArray(total).also { gxM.get(0, 0, it) }
        val gy = FloatArray(total).also { gyM.get(0, 0, it) }
        val near = ByteArray(total).also { nearM.get(0, 0, it) }
        gxM.release(); gyM.release(); nearM.release()

        val mag = FloatArray(total)
        var nNear = 0
        for (i in 0 until total) {
            mag[i] = sqrt(gx[i] * gx[i] + gy[i] * gy[i])
            if (near[i].toInt() != 0) nNear++
        }
        if (nNear < 500) return 0.0
        val mags = FloatArray(nNear)
        var q = 0
        for (i in 0 until total) if (near[i].toInt() != 0) mags[q++] = mag[i]
        val thr = percentile(mags, 60.0)
        val angA = DoubleArray(nNear)
        val wtA = DoubleArray(nNear)
        var nSel = 0
        for (i in 0 until total) {
            if (near[i].toInt() != 0 && mag[i] > thr && abs(gx[i]) > 1e-3f) {
                angA[nSel] = Math.toDegrees(atan((gy[i] / gx[i]).toDouble()))
                wtA[nSel] = mag[i].toDouble()
                nSel++
            }
        }
        if (nSel < 500) return 0.0
        var est = 0.0
        for (win in doubleArrayOf(15.0, 6.0)) {
            val nb = (2 * win / 0.1).roundToInt()
            val histD = DoubleArray(nb) // numpy bincount 처럼 double 로 누적
            for (k in 0 until nSel) {
                if (abs(angA[k] - est) < win) {
                    val idx = ((angA[k] - (est - win)) / 0.1).toInt().coerceIn(0, nb - 1)
                    histD[idx] += wtA[k]
                }
            }
            val hist = FloatArray(nb) { histD[it].toFloat() }
            val hm = Mat(1, nb, CvType.CV_32F)
            hm.put(0, 0, hist)
            Imgproc.GaussianBlur(hm, hm, Size(0.0, 0.0), 4.0, 0.01, Core.BORDER_REPLICATE)
            hm.get(0, 0, hist)
            hm.release()
            var bi = 0
            for (k in 1 until nb) if (hist[k] > hist[bi]) bi = k
            est = est - win + (bi + 0.5) * 0.1
        }
        return -est
    }

    // ================================================================= 글줄 곡선

    private class Line(val poly: Poly, val kx: DoubleArray, val ky: DoubleArray, val x0: Double, val x1: Double, val t: Double)

    private fun fitLine(xs: DoubleArray, ys: DoubleArray, w: Int, ch: Double): Line? {
        val span = xs.max() - xs.min()
        val deg = if (span > w * 0.5) 3 else if (span > w * 0.25) 2 else 1
        var keep = BooleanArray(xs.size) { true }
        var res = DoubleArray(xs.size)
        var poly: Poly? = null
        for (iter in 0 until 2) {
            val kk = keep
            val idx = kk.indices.filter { kk[it] }
            if (idx.size < deg + 3) return null
            val p = polyfit(DoubleArray(idx.size) { xs[idx[it]] }, DoubleArray(idx.size) { ys[idx[it]] }, deg) ?: return null
            poly = p
            val r = DoubleArray(xs.size) { ys[it] - p.eval(xs[it]) }
            res = r
            val tol = max(0.35 * ch, 1.0)
            keep = BooleanArray(xs.size) { abs(r[it]) < tol }
        }
        val kk = keep
        val rr = res
        val kept = kk.indices.filter { kk[it] }
        if (kept.isEmpty()) return null
        val rms = sqrt(kept.sumOf { rr[it] * rr[it] } / kept.size)
        if (rms > 0.25 * ch) return null
        val kx = DoubleArray(kept.size) { xs[kept[it]] }
        var ky = DoubleArray(kept.size) { ys[kept[it]] }
        if (ky.size >= 3) {
            val n = ky.size
            val src = ky
            ky = DoubleArray(n) { j -> (src[max(j - 1, 0)] + 2 * src[j] + src[min(j + 1, n - 1)]) / 4.0 }
        }
        return Line(poly!!, kx, ky, kx.min(), kx.max(), median(ky))
    }

    /** 글줄 곡선의 x 위치 높이. 범위 안은 표본 보간, 바깥은 끝 기울기로 연장 */
    private fun lineY(l: Line, x: Double): Double {
        if (x < l.x0) return l.ky[0] + l.poly.deriv(l.x0) * (x - l.x0)
        if (x > l.x1) return l.ky[l.ky.size - 1] + l.poly.deriv(l.x1) * (x - l.x1)
        return interp(x, l.kx, l.ky)
    }

    // ================================================================= 세로 이동 곡면

    private class Surface(val coef: DoubleArray, val x0: Double, val x1: Double, val y0: Double, val y1: Double)

    private fun basis(xn: Double, yn: Double, out: DoubleArray) {
        var k = 0
        var yp = 1.0
        for (j in 0 until 3) {
            var xp = 1.0
            for (i in 0 until 4) {
                out[k++] = xp * yp
                xp *= xn
            }
            yp *= yn
        }
    }

    private fun fitSurface(X: DoubleArray, Y: DoubleArray, D: DoubleArray, w: Int, h: Int, ch: Double): Surface? {
        if (X.size < 24) return null
        val n = X.size
        val A = Array(n) { DoubleArray(12) }
        for (k in 0 until n) basis((X[k] - w / 2.0) / (w / 2.0), (Y[k] - h / 2.0) / (h / 2.0), A[k])
        var keep = BooleanArray(n) { true }
        var coef: DoubleArray? = null
        for (iter in 0 until 3) {
            val kk = keep
            if (kk.count { b -> b } < 24) break
            val c = lstsqSvd(A, D, kk) ?: break
            coef = c
            val r = DoubleArray(n) { k -> D[k] - dot(A[k], c) }
            val absKept = kk.indices.filter { kk[it] }.map { abs(r[it]) }.toDoubleArray()
            val thr = max(1.5 * median(absKept) * 1.4826 * 2, 0.3 * ch)
            keep = BooleanArray(n) { abs(r[it]) < thr }
        }
        val c = coef ?: return null
        return Surface(c, X.min(), X.max(), Y.min(), Y.max())
    }

    private fun evalSurface(s: Surface?, x: Double, y: Double, w: Int, h: Int, tmp: DoubleArray): Double {
        if (s == null) return 0.0
        val xc = x.coerceIn(s.x0, s.x1)
        val yc = y.coerceIn(s.y0, s.y1)
        basis((xc - w / 2.0) / (w / 2.0), (yc - h / 2.0) / (h / 2.0), tmp)
        return dot(tmp, s.coef)
    }

    // ================================================================= 좌우 여백선

    private class Margin(val poly: Poly, val t0: Double, val t1: Double) {
        fun at(v: Double) = poly.eval(v.coerceIn(t0, t1))
    }

    /** 여백선 x = m(T). 직선으로 이상치를 거른 뒤 곡선(2~3차)으로 다시 맞춘다. 모든 두 줄 조합 시험 → 결정적 */
    private fun fitMargin(T: DoubleArray, X: DoubleArray, ch: Double, h: Int): Margin? {
        val n = T.size
        if (n < 5) return null
        var best: BooleanArray? = null
        var bestCnt = -1
        val tol = 1.2 * ch
        for (i in 0 until n) for (j in i + 1 until n) {
            if (abs(T[i] - T[j]) < h * 0.1) continue
            val a = (X[j] - X[i]) / (T[j] - T[i])
            val b = X[i] - a * T[i]
            val cand = BooleanArray(n) { abs(X[it] - (a * T[it] + b)) < tol }
            val c = cand.count { v -> v }
            if (c > bestCnt) {
                best = cand; bestCnt = c
            }
        }
        var inl = best ?: return null
        var poly: Poly? = null
        for (iter in 0 until 2) {
            val cur = inl
            val idx = cur.indices.filter { cur[it] }
            if (idx.size < 5) return null
            val deg = if (idx.size >= 12) 3 else if (idx.size >= 8) 2 else 1
            val p = polyfit(DoubleArray(idx.size) { T[idx[it]] }, DoubleArray(idx.size) { X[idx[it]] }, deg) ?: return null
            poly = p
            inl = BooleanArray(n) { abs(X[it] - p.eval(T[it])) < 0.6 * ch }
        }
        val fin = inl
        val idx = fin.indices.filter { fin[it] }
        if (idx.size < max(5.0, 0.3 * n)) return null
        val tIn = idx.map { T[it] }
        if (tIn.max() - tIn.min() < 0.3 * h) return null
        return Margin(poly!!, tIn.min(), tIn.max())
    }

    // ================================================================= 본체

    private fun core(gray: Mat, clean: Mat, chIn: Double, phi: Double): Maps {
        val h = gray.rows()
        val w = gray.cols()
        val cx = (w - 1) / 2.0
        val cy = (h - 1) / 2.0
        val t = Math.toRadians(phi)
        // 출력(u,v) → 입력(x,y) = c + R(phi)·((u,v) − c)
        val m00 = cos(t); val m01 = -sin(t); val m10 = sin(t); val m11 = cos(t)
        val m02 = cx - (m00 * cx + m01 * cy)
        val m12 = cy - (m10 * cx + m11 * cy)
        val M = Mat(2, 3, CvType.CV_64F)
        M.put(0, 0, m00, m01, m02, m10, m11, m12)

        // 기울기를 바로잡은 이미지 D 에서 글줄을 찾고, 표본점을 원래 좌표로 되돌린다 (D 는 탐지에만 사용)
        val D = Mat()
        Imgproc.warpAffine(gray, D, M, Size(w.toDouble(), h.toDouble()), Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP, Core.BORDER_REPLICATE)
        M.release()
        val (raw0, ch2) = detectLines(D, 0.05)
        val ch = ch2 ?: chIn
        val raw = raw0 + detectRules(D, ch)
        D.release()
        val lines = ArrayList<Line>()
        for (s in raw) {
            val n = s.xs.size
            val xo = DoubleArray(n) { m00 * s.xs[it] + m01 * s.ys[it] + m02 }
            val yo = DoubleArray(n) { m10 * s.xs[it] + m11 * s.ys[it] + m12 }
            val order = (0 until n).sortedBy { xo[it] } // 안정 정렬
            val l = fitLine(DoubleArray(n) { xo[order[it]] }, DoubleArray(n) { yo[order[it]] }, w, ch) ?: continue
            lines.add(l)
        }
        lines.sortBy { it.t }

        // ---- 세로 이동량 지도 (거친 격자, 칸 = 글자 높이): d = 매끈한 전체 곡면 S + 글줄 근처 잔차 R
        val gstep = max(4, ch.toInt())
        val gw = w / gstep + 2
        val gh = h / gstep + 2
        val dfield = FloatArray(gh * gw)
        if (lines.size >= 3) {
            val X = ArrayList<Double>()
            val Y = ArrayList<Double>()
            val Dv = ArrayList<Double>()
            for (l in lines) {
                val c0 = ceil(l.x0 / gstep).toInt()
                val c1 = floor(l.x1 / gstep).toInt()
                for (ci in c0..c1) {
                    val x = (ci * gstep).toDouble()
                    X.add(x); Y.add(l.t); Dv.add(lineY(l, x) - l.t)
                }
            }
            val xa = X.toDoubleArray()
            val ya = Y.toDoubleArray()
            val da = Dv.toDoubleArray()
            val surf = fitSurface(xa, ya, da, w, h, ch)
            val tmp = DoubleArray(12)
            val num = FloatArray(gh * gw)
            val den = FloatArray(gh * gw)
            val numD = DoubleArray(gh * gw)
            val denD = DoubleArray(gh * gw)
            for (k in xa.indices) {
                val res = da[k] - evalSurface(surf, xa[k], ya[k], w, h, tmp).toFloat()
                val cxi = (xa[k] / gstep).roundToInt().coerceIn(0, gw - 1)
                val fy = ya[k] / gstep
                val y0 = floor(fy).toInt().coerceIn(0, gh - 1)
                val y1 = (y0 + 1).coerceIn(0, gh - 1)
                val wy1 = fy - floor(fy)
                numD[y0 * gw + cxi] += res * (1 - wy1); denD[y0 * gw + cxi] += 1 - wy1
                numD[y1 * gw + cxi] += res * wy1; denD[y1 * gw + cxi] += wy1
            }
            for (k in num.indices) {
                num[k] = numD[k].toFloat(); den[k] = denD[k].toFloat()
            }
            val numM = Mat(gh, gw, CvType.CV_32F).also { it.put(0, 0, num) }
            val denM = Mat(gh, gw, CvType.CV_32F).also { it.put(0, 0, den) }
            Imgproc.GaussianBlur(numM, numM, Size(0.0, 0.0), 1.2, 1.0)
            Imgproc.GaussianBlur(denM, denM, Size(0.0, 0.0), 1.2, 1.0)
            numM.get(0, 0, num); denM.get(0, 0, den)
            numM.release(); denM.release()
            for (r in 0 until gh) for (c in 0 until gw) {
                val s = evalSurface(surf, (c * gstep).toDouble(), (r * gstep).toDouble(), w, h, tmp).toFloat()
                val k = r * gw + c
                dfield[k] = s + num[k] / (den[k] + 0.02f)
            }
        }

        // ---- 좌우 여백선
        var left: Margin? = null
        var right: Margin? = null
        if (lines.size >= 5) {
            val T = DoubleArray(lines.size) { lines[it].t }
            left = fitMargin(T, DoubleArray(lines.size) { lines[it].x0 }, ch, h)
            right = fitMargin(T, DoubleArray(lines.size) { lines[it].x1 }, ch, h)
        }

        // ---- 가로 보정: 여백선이 있으면 여백선으로, 없으면 글자 세로획 각도로
        val lm = left
        val rm = right
        var shear = 0.0
        if (lm == null) {
            val stem = stemAngle(gray, clean)
            if (abs(stem) >= 0.5) shear = tan(Math.toRadians(stem)) // 측정 오차 수준은 무시
        }
        val l0 = lm?.at(h / 2.0) ?: 0.0
        val r0 = rm?.at(h / 2.0) ?: 0.0
        // 행 v 에서 xv = a(v) + b(v)·u 꼴
        fun rowCoef(v: Double): DoubleArray {
            if (lm != null) {
                val lv = lm.at(v)
                if (rm != null) {
                    val ratio = ((rm.at(v) - lv) / max(r0 - l0, 1.0)).coerceIn(0.9, 1.1)
                    return doubleArrayOf(lv - l0 * ratio, ratio)
                }
                return doubleArrayOf(lv - l0, 1.0)
            }
            return doubleArrayOf(shear * (v - cy), 1.0)
        }

        fun dAt(x: Double, v: Double): Double { // dfield 이중선형 보간(가장자리 반복)
            val gxq = (x / gstep).coerceIn(0.0, gw - 1.0)
            val gyq = (v / gstep).coerceIn(0.0, gh - 1.0)
            val x0 = floor(gxq).toInt().coerceAtMost(gw - 2).coerceAtLeast(0)
            val y0 = floor(gyq).toInt().coerceAtMost(gh - 2).coerceAtLeast(0)
            val fx = gxq - x0
            val fy = gyq - y0
            val a = dfield[y0 * gw + x0] * (1 - fx) + dfield[y0 * gw + x0 + 1] * fx
            val b = dfield[(y0 + 1) * gw + x0] * (1 - fx) + dfield[(y0 + 1) * gw + x0 + 1] * fx
            return a * (1 - fy) + b * fy
        }

        // ---- 보정으로 화면 밖으로 밀려나는 내용이 없도록 캔버스를 넓힘 (머리글·쪽번호 잘림 방지)
        val cTop = rowCoef(0.0)
        val cBot = rowCoef(h - 1.0)
        var top = -Double.MAX_VALUE
        var bot = Double.MAX_VALUE
        for (u in 0 until w) {
            top = max(top, dAt(cTop[0] + cTop[1] * u, 0.0))
            bot = min(bot, dAt(cBot[0] + cBot[1] * u, h - 1.0))
        }
        var leftMost = -Double.MAX_VALUE
        var rightMost = -Double.MAX_VALUE
        for (v in 0 until h) {
            val cf = rowCoef(v.toDouble())
            leftMost = max(leftMost, cf[0])
            rightMost = max(rightMost, (w - 1.0) - (cf[0] + cf[1] * (w - 1.0)))
        }
        val padT = min(ceil(max(0.0, top)).toInt(), (h * 0.1).toInt())
        val padB = min(ceil(max(0.0, -bot)).toInt(), (h * 0.1).toInt())
        val padL = min(ceil(max(0.0, leftMost)).toInt(), (w * 0.1).toInt())
        val padR = min(ceil(max(0.0, rightMost)).toInt(), (w * 0.1).toInt())
        val W2 = w + padL + padR
        val H2 = h + padT + padB

        // ---- 출력 픽셀 (u, v) = (i − padL, j − padT) → 입력 (xv, v + d(xv, v))
        val buf = FloatArray(H2 * W2)
        for (j in 0 until H2) {
            val cf = rowCoef((j - padT).toDouble())
            for (i in 0 until W2) buf[j * W2 + i] = (cf[0] + cf[1] * (i - padL)).toFloat()
        }
        val mapX = Mat(H2, W2, CvType.CV_32F).also { it.put(0, 0, buf) }
        for (j in 0 until H2) {
            val v = (j - padT).toFloat()
            for (i in 0 until W2) buf[j * W2 + i] = v
        }
        val vM = Mat(H2, W2, CvType.CV_32F).also { it.put(0, 0, buf) }
        val gxM = Mat()
        val gyM = Mat()
        Core.multiply(mapX, Scalar(1.0 / gstep), gxM)
        Core.multiply(vM, Scalar(1.0 / gstep), gyM)
        val dM = Mat(gh, gw, CvType.CV_32F).also { it.put(0, 0, dfield) }
        val dmap = Mat()
        Imgproc.remap(dM, dmap, gxM, gyM, Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE, Scalar(0.0))
        dM.release(); gxM.release(); gyM.release()
        val mapY = Mat()
        Core.add(vM, dmap, mapY)
        vM.release(); dmap.release()
        return Maps(mapX, mapY)
    }

    // ================================================================= 수치 도구

    /** 다항식. 수치 안정을 위해 x 를 (x − xm) / xs 로 정규화해서 맞춘다 */
    class Poly(private val c: DoubleArray, private val xm: Double, private val xs: Double) {
        fun eval(x: Double): Double {
            val t = (x - xm) / xs
            var r = 0.0
            for (i in c.indices.reversed()) r = r * t + c[i]
            return r
        }

        fun deriv(x: Double): Double {
            val t = (x - xm) / xs
            var r = 0.0
            for (i in c.indices.reversed()) if (i >= 1) r = r * t + i * c[i]
            return r / xs
        }
    }

    fun polyfit(x: DoubleArray, y: DoubleArray, deg: Int): Poly? {
        val n = x.size
        if (n <= deg) return null
        val xm = x.average()
        var half = 0.0
        for (v in x) half = max(half, abs(v - xm))
        val xs = if (half < 1e-9) 1.0 else half
        val m = deg + 1
        val A = Array(n) { k ->
            val t = (x[k] - xm) / xs
            DoubleArray(m).also { row ->
                var p = 1.0
                for (i in 0 until m) {
                    row[i] = p; p *= t
                }
            }
        }
        val c = leastSquares(A, y, null) ?: return null
        return Poly(c, xm, xs)
    }

    /** 최소제곱: (AᵀA) c = Aᵀy, keep 이 있으면 그 행만 사용 */
    private fun leastSquares(A: Array<DoubleArray>, y: DoubleArray, keep: BooleanArray?): DoubleArray? {
        val m = A[0].size
        val ata = Array(m) { DoubleArray(m) }
        val aty = DoubleArray(m)
        for (k in A.indices) {
            if (keep != null && !keep[k]) continue
            val row = A[k]
            for (i in 0 until m) {
                aty[i] += row[i] * y[k]
                for (j in 0 until m) ata[i][j] += row[i] * row[j]
            }
        }
        return solve(ata, aty)
    }

    /** 최소제곱을 SVD 로 직접 풂(정규방정식보다 수치적으로 안정, numpy.linalg.lstsq 와 같은 결과) */
    private fun lstsqSvd(A: Array<DoubleArray>, y: DoubleArray, keep: BooleanArray): DoubleArray? {
        val m = A[0].size
        val rows = keep.indices.filter { keep[it] }
        if (rows.size < m) return null
        val a = Mat(rows.size, m, CvType.CV_64F)
        val b = Mat(rows.size, 1, CvType.CV_64F)
        val flat = DoubleArray(rows.size * m)
        val bv = DoubleArray(rows.size)
        for ((r, k) in rows.withIndex()) {
            System.arraycopy(A[k], 0, flat, r * m, m)
            bv[r] = y[k]
        }
        a.put(0, 0, *flat)
        b.put(0, 0, *bv)
        val x = Mat()
        val ok = Core.solve(a, b, x, Core.DECOMP_SVD)
        val out = DoubleArray(m)
        if (ok) x.get(0, 0, out)
        a.release(); b.release(); x.release()
        return if (ok) out else null
    }

    /** 부분 피벗 가우스 소거 */
    private fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
        val n = b.size
        val m = Array(n) { a[it].copyOf() }
        val v = b.copyOf()
        for (col in 0 until n) {
            var piv = col
            for (r in col + 1 until n) if (abs(m[r][col]) > abs(m[piv][col])) piv = r
            if (abs(m[piv][col]) < 1e-12) return null
            val tmp = m[col]; m[col] = m[piv]; m[piv] = tmp
            val tv = v[col]; v[col] = v[piv]; v[piv] = tv
            for (r in col + 1 until n) {
                val f = m[r][col] / m[col][col]
                if (f == 0.0) continue
                for (c in col until n) m[r][c] -= f * m[col][c]
                v[r] -= f * v[col]
            }
        }
        val x = DoubleArray(n)
        for (r in n - 1 downTo 0) {
            var s = v[r]
            for (c in r + 1 until n) s -= m[r][c] * x[c]
            x[r] = s / m[r][r]
        }
        return x
    }

    private fun dot(a: DoubleArray, b: DoubleArray): Double {
        var s = 0.0
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    /** 오름차순 xs 에 대한 선형 보간 (numpy.interp 와 같음) */
    private fun interp(x: Double, xs: DoubleArray, ys: DoubleArray): Double {
        if (x <= xs[0]) return ys[0]
        if (x >= xs[xs.size - 1]) return ys[ys.size - 1]
        var lo = 0
        var hi = xs.size - 1
        while (hi - lo > 1) {
            val mid = (lo + hi) / 2
            if (xs[mid] <= x) lo = mid else hi = mid
        }
        val dx = xs[hi] - xs[lo]
        return if (dx == 0.0) ys[hi] else ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / dx
    }

    private fun median(v: DoubleArray): Double = BookScanEngine.median(v)
    private fun percentile(v: FloatArray, p: Double): Float = BookScanEngine.percentile(v, p)
}
