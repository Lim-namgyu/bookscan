package com.bookscan.app.camera

import android.graphics.RectF
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.bookscan.app.engine.BookRegion
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * 촬영 준비 상태 판단 + 자동 촬영.
 *
 * 프리뷰 프레임(RGBA)을 보고 "지금 찍으면 좋은 스캔이 나오는지"를 매 프레임 판단한다.
 *  - 책이 화면 안에 다 들어와 있고, 충분히 크고, 가운데에 있는지   (책 영역 모델, 약 0.4초마다)
 *  - 폰이 책과 나란한지(기울기)                                    (가속도 센서, 바깥에서 넣어줌)
 *  - 너무 어둡거나 반사로 하얗게 날아가지 않았는지
 *  - 손이 멈춰 있는지 (직전 프레임과 차이)
 *  - 초점이 맞았는지 (글자 가장자리의 또렷함)
 * 모두 통과하면 READY. 자동 촬영이 켜져 있으면 READY 가 잠시 유지되고 지난번 찍은 장면과 달라졌을 때 찍는다.
 */
class CaptureReadiness(private val regionLoader: () -> BookRegion?) : ImageAnalysis.Analyzer {

    enum class Issue {
        NO_BOOK, TOO_SMALL, CUT_OFF, OFF_CENTER, TILTED, DARK, BRIGHT, MOVING, BLURRY, NONE
    }

    /**
     * ready: 지금 찍어도 좋음. issue: 가장 먼저 고칠 것. box: 책 영역(화면을 바로 세운 기준, 0~1 비율), 없으면 null.
     * focusStuck: 멈춰 있는데도 초점이 오래 안 맞음(탭해서 초점 맞추기를 안내).
     */
    class Status(val ready: Boolean, val issue: Issue, val box: RectF?, val focusStuck: Boolean)

    /** 폰 기울기(도). 0 = 책(바닥)과 평행. 센서에서 넣어준다 */
    @Volatile var tiltDeg: Float = 0f

    @Volatile var autoEnabled: Boolean = false

    /** 상태 변화 알림 (분석 스레드에서 호출됨) */
    @Volatile var onStatus: ((Status) -> Unit)? = null

    /** 자동 촬영 신호 (분석 스레드에서 호출됨) */
    @Volatile var onTrigger: (() -> Unit)? = null

    private var region: BookRegion? = null
    private var regionTried = false

    // 움직임 판단용 작은 밝기 격자
    private var prev: FloatArray? = null
    @Volatile private var lastShot: FloatArray? = null
    private var cooldownUntil = 0L

    // 책 영역 판단(모델)은 비싸서 간격을 두고 돌리고, 그 사이에는 마지막 결과를 쓴다
    private var lastDetectAt = 0L
    private var detIssue: Issue = Issue.NO_BOOK
    private var detBox: RectF? = null            // 프레임 좌표(0~1), 회전 전
    private var detBoxPx: IntArray? = null       // 프레임 픽셀 좌표 x0, y0, x1, y1

    private var readyStreak = 0
    private var stillSince = 0L
    private var lastStatus: Status? = null

    override fun analyze(image: ImageProxy) {
        try {
            val now = System.currentTimeMillis()
            val w = image.width
            val h = image.height
            val plane = image.planes[0]
            val buf = plane.buffer
            val rowStride = plane.rowStride
            val pixStride = plane.pixelStride

            // ---- 밝기 격자(움직임) : 작은 격자
            val cur = FloatArray(GW * GH)
            var sum = 0f
            for (gy in 0 until GH) {
                val y = (gy * h + h / 2) / GH
                for (gx in 0 until GW) {
                    val x = (gx * w + w / 2) / GW
                    val v = luma(buf, y * rowStride + x * pixStride)
                    cur[gy * GW + gx] = v
                    sum += v
                }
            }
            val mean = (sum / cur.size).coerceAtLeast(1f)
            for (i in cur.indices) cur[i] /= mean
            val p = prev
            prev = cur
            val motion = if (p == null) 100f else diff(cur, p)

            // ---- 책 영역 (간격을 두고)
            if (now - lastDetectAt >= DETECT_MS) {
                lastDetectAt = now
                detect(buf, w, h, rowStride, pixStride)
            }

            // ---- 밝기·또렷함: 책 영역 안(없으면 가운데 80%)에서, 가로 2픽셀 간격으로 샘플
            val bx = detBoxPx ?: intArrayOf((w * 0.1).toInt(), (h * 0.1).toInt(), (w * 0.9).toInt(), (h * 0.9).toInt())
            val (bright, clipped, sharp) = measure(buf, rowStride, pixStride, bx)

            // ---- 판단 (먼저 고쳐야 할 것부터)
            val issue = when {
                detIssue != Issue.NONE -> detIssue
                tiltDeg > TILT_MAX -> Issue.TILTED
                bright < DARK_T -> Issue.DARK
                clipped > CLIP_T -> Issue.BRIGHT
                motion > STILL_T -> Issue.MOVING
                sharp < SHARP_T -> Issue.BLURRY
                else -> Issue.NONE
            }
            if (motion > STILL_T) stillSince = now
            val focusStuck = issue == Issue.BLURRY && now - stillSince > FOCUS_STUCK_MS
            val ready = issue == Issue.NONE
            readyStreak = if (ready) readyStreak + 1 else 0

            val box = detBox?.let { upright(it, image.imageInfo.rotationDegrees) }
            val st = Status(ready, issue, box, focusStuck)
            val prevSt = lastStatus
            if (prevSt == null || prevSt.ready != st.ready || prevSt.issue != st.issue || prevSt.focusStuck != st.focusStuck ||
                boxMoved(prevSt.box, st.box)
            ) {
                lastStatus = st
                onStatus?.invoke(st)
            }

            // ---- 자동 촬영: 준비 상태가 잠시 유지되고, 지난번 찍은 장면과 달라졌을 때
            if (autoEnabled && ready && readyStreak >= READY_FRAMES && now >= cooldownUntil) {
                val shot = lastShot
                if (shot == null || diff(cur, shot) > CHANGE_T) {
                    lastShot = cur
                    cooldownUntil = now + COOLDOWN_MS
                    readyStreak = 0
                    onTrigger?.invoke()
                }
            }
        } catch (_: Throwable) {
            // 분석은 안내용이므로 실패해도 촬영에는 영향 없음
        } finally {
            image.close()
        }
    }

    /** 수동 촬영 직후 호출 → 같은 장면을 자동으로 또 찍지 않게 */
    fun markCaptured() {
        lastShot = prev
        cooldownUntil = System.currentTimeMillis() + COOLDOWN_MS
    }

    fun close() {
        region?.close()
        region = null
    }

    // ------------------------------------------------------------------ 책 영역

    private fun detect(buf: java.nio.ByteBuffer, w: Int, h: Int, rowStride: Int, pixStride: Int) {
        if (!regionTried) {
            regionTried = true
            region = runCatching { regionLoader() }.getOrNull()
        }
        val reg = region
        if (reg == null) { // 모델이 없으면 책 위치는 판단하지 않음
            detIssue = Issue.NONE; detBox = null; detBoxPx = null
            return
        }
        // RGBA 프레임 → RGB Mat (긴 변 DET_SIDE 로 줄여서)
        val s = min(1.0, DET_SIDE / max(w, h))
        val dw = max(16, (w * s).toInt())
        val dh = max(16, (h * s).toInt())
        val rgb = ByteArray(dw * dh * 3)
        var k = 0
        for (y in 0 until dh) {
            val sy = (y * h + h / 2) / dh
            for (x in 0 until dw) {
                val sx = (x * w + w / 2) / dw
                val o = sy * rowStride + sx * pixStride
                rgb[k] = buf.get(o); rgb[k + 1] = buf.get(o + 1); rgb[k + 2] = buf.get(o + 2)
                k += 3
            }
        }
        val m = Mat(dh, dw, CvType.CV_8UC3).also { it.put(0, 0, rgb) }
        val res = reg.mask(m)
        m.release()
        if (res == null) {
            detIssue = when (reg.lastReason) {
                BookRegion.REASON_FILLS -> Issue.CUT_OFF
                BookRegion.REASON_SMALL -> Issue.TOO_SMALL
                else -> Issue.NO_BOOK
            }
            detBox = null; detBoxPx = null
            return
        }
        val (mask, box) = res
        val mb = ByteArray(dw * dh).also { mask.get(0, 0, it) }
        mask.release()
        // 네 변에 닿은 비율: 변 가까이(2%) 띠 안에서 마스크가 차지하는 길이 비율
        val ex = max(1, (dw * 0.02).toInt())
        val ey = max(1, (dh * 0.02).toInt())
        fun touchCol(x0: Int, x1: Int): Float {
            var n = 0
            for (y in 0 until dh) { var hit = false; for (x in x0 until x1) if (mb[y * dw + x].toInt() != 0) { hit = true; break }; if (hit) n++ }
            return n.toFloat() / dh
        }
        fun touchRow(y0: Int, y1: Int): Float {
            var n = 0
            for (x in 0 until dw) { var hit = false; for (y in y0 until y1) if (mb[y * dw + x].toInt() != 0) { hit = true; break }; if (hit) n++ }
            return n.toFloat() / dw
        }
        val touch = max(max(touchCol(0, ex), touchCol(dw - ex, dw)), max(touchRow(0, ey), touchRow(dh - ey, dh)))
        var area = 0
        for (b in mb) if (b.toInt() != 0) area++
        val areaFrac = area.toFloat() / (dw * dh)
        val cx = (box.x + box.width / 2f) / dw
        val cy = (box.y + box.height / 2f) / dh
        detBox = RectF(box.x.toFloat() / dw, box.y.toFloat() / dh, (box.x + box.width).toFloat() / dw, (box.y + box.height).toFloat() / dh)
        detBoxPx = intArrayOf(
            (box.x / s).toInt().coerceIn(0, w - 1), (box.y / s).toInt().coerceIn(0, h - 1),
            ((box.x + box.width) / s).toInt().coerceIn(1, w), ((box.y + box.height) / s).toInt().coerceIn(1, h)
        )
        detIssue = when {
            touch > TOUCH_T -> Issue.CUT_OFF
            areaFrac < AREA_MIN -> Issue.TOO_SMALL
            abs(cx - 0.5f) > CENTER_T || abs(cy - 0.5f) > CENTER_T -> Issue.OFF_CENTER
            else -> Issue.NONE
        }
    }

    // ------------------------------------------------------------------ 밝기·또렷함

    /** (평균 밝기, 날아간 비율, 또렷함) — 영역 안에서 2픽셀 간격 샘플 */
    private fun measure(buf: java.nio.ByteBuffer, rowStride: Int, pixStride: Int, bx: IntArray): Triple<Float, Float, Float> {
        val x0 = bx[0]; val y0 = bx[1]; val x1 = bx[2]; val y1 = bx[3]
        val sw = max(8, (x1 - x0) / 2)
        val sh = max(8, (y1 - y0) / 2)
        val g = FloatArray(sw * sh)
        var sum = 0f
        var clip = 0
        for (y in 0 until sh) {
            val sy = min(y1 - 1, y0 + y * 2)
            for (x in 0 until sw) {
                val sx = min(x1 - 1, x0 + x * 2)
                val v = luma(buf, sy * rowStride + sx * pixStride)
                g[y * sw + x] = v
                sum += v
                if (v >= 250f) clip++
            }
        }
        val mean = sum / g.size
        // 대비: 5%~95% 밝기 폭 (히스토그램)
        val hist = IntArray(256)
        for (v in g) hist[v.toInt().coerceIn(0, 255)]++
        var acc = 0; var p5 = 0; var p95 = 255
        for (v in 0 until 256) { acc += hist[v]; if (acc >= g.size * 0.05) { p5 = v; break } }
        acc = 0
        for (v in 255 downTo 0) { acc += hist[v]; if (acc >= g.size * 0.05) { p95 = v; break } }
        val contrast = max(10f, (p95 - p5).toFloat())
        // 라플라시안 절대값의 상위 10% 평균 / 대비 : 글자 가장자리가 또렷할수록 큼, 노출과 무관
        val lap = FloatArray((sw - 2) * (sh - 2))
        var k = 0
        for (y in 1 until sh - 1) for (x in 1 until sw - 1) {
            val c = g[y * sw + x]
            lap[k++] = abs(g[(y - 1) * sw + x] + g[(y + 1) * sw + x] + g[y * sw + x - 1] + g[y * sw + x + 1] - 4f * c)
        }
        lap.sort()
        val n10 = max(1, lap.size / 10)
        var top = 0f
        for (i in lap.size - n10 until lap.size) top += lap[i]
        val sharp = top / n10 / contrast
        return Triple(mean, clip.toFloat() / g.size, sharp)
    }

    private fun luma(buf: java.nio.ByteBuffer, o: Int): Float {
        val r = buf.get(o).toInt() and 0xFF
        val g = buf.get(o + 1).toInt() and 0xFF
        val b = buf.get(o + 2).toInt() and 0xFF
        return ((r * 77 + g * 150 + b * 29) shr 8).toFloat()
    }

    private fun diff(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += abs(a[i] - b[i])
        return s / a.size * 100f
    }

    /** 프레임 좌표(0~1)의 상자를 화면을 바로 세운 좌표로 (rotation = 바로 세우기 위해 시계 방향으로 돌릴 각도) */
    private fun upright(r: RectF, rotation: Int): RectF = when (rotation) {
        90 -> RectF(1f - r.bottom, r.left, 1f - r.top, r.right)
        180 -> RectF(1f - r.right, 1f - r.bottom, 1f - r.left, 1f - r.top)
        270 -> RectF(r.top, 1f - r.right, r.bottom, 1f - r.left)
        else -> RectF(r)
    }

    private fun boxMoved(a: RectF?, b: RectF?): Boolean {
        if (a == null || b == null) return a !== b
        return abs(a.left - b.left) > 0.02f || abs(a.top - b.top) > 0.02f || abs(a.right - b.right) > 0.02f || abs(a.bottom - b.bottom) > 0.02f
    }

    companion object {
        private const val GW = 64
        private const val GH = 48
        private const val DET_SIDE = 480.0
        private const val DETECT_MS = 400L

        // 아래 수치는 실제 기기에서 써보며 조정할 값
        const val STILL_T = 1.5f        // 직전 프레임과 평균 밝기 차(%) 가 이보다 작으면 '멈춤'
        const val SHARP_T = 0.7f        // 또렷함 기준 (PC 시험: 또렷한 사진 1.1~5.7, 많이 흐린 사진 0.1~0.45)
        const val DARK_T = 60f          // 책 영역 평균 밝기(0~255)가 이보다 어두우면 '어두움'
        const val CLIP_T = 0.08f        // 밝기 250 이상(날아감) 비율이 이보다 크면 '너무 밝음'
        const val TILT_MAX = 20f        // 폰 기울기(도)
        const val AREA_MIN = 0.22f      // 책이 화면에서 차지하는 비율 최소
        const val TOUCH_T = 0.25f       // 책이 화면 한 변에 이 비율 이상 닿아 있으면 '잘림'
        const val CENTER_T = 0.18f      // 책 중심이 화면 중심에서 이보다 벗어나면 '가운데로'
        const val READY_FRAMES = 4      // 준비 상태가 이만큼 연속되면 자동 촬영
        const val CHANGE_T = 4f         // 지난 촬영 장면과 이만큼은 달라야 새 페이지로 인정
        const val COOLDOWN_MS = 1200L
        const val FOCUS_STUCK_MS = 2500L
    }
}
