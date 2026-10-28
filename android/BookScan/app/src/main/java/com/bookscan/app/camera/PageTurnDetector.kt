package com.bookscan.app.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlin.math.abs

/**
 * 페이지 넘김 자동 촬영.
 * 프리뷰 프레임을 64x48 밝기 격자로 줄여 직전 프레임과 비교한다.
 *  대기 → (큰 움직임: 페이지 넘김) → 움직임 → (잠잠해짐) → 고정 중 → (STILL_MS 동안 유지) → 촬영
 * 마지막으로 찍은 장면과 충분히 달라졌을 때만 찍어서, 손만 움찔한 경우의 중복 촬영을 막는다.
 */
class PageTurnDetector : ImageAnalysis.Analyzer {

    enum class State { WAITING, MOVING, STEADYING }

    @Volatile var enabled: Boolean = false
        set(value) {
            field = value
            state = State.WAITING
        }

    /** 촬영 신호 (분석 스레드에서 호출됨) */
    @Volatile var onTrigger: (() -> Unit)? = null

    /** 상태 변화 알림 (분석 스레드에서 호출됨) */
    @Volatile var onState: ((State) -> Unit)? = null

    private var prev: FloatArray? = null
    @Volatile private var lastShot: FloatArray? = null
    private var state = State.WAITING
        set(value) {
            if (field != value) {
                field = value
                onState?.invoke(value)
            }
        }
    private var stillSince = 0L
    private var cooldownUntil = 0L

    override fun analyze(image: ImageProxy) {
        try {
            val cur = thumbnail(image)
            val now = System.currentTimeMillis()
            val p = prev
            prev = cur
            if (!enabled || p == null || now < cooldownUntil) return
            val motion = diff(cur, p)
            when (state) {
                State.WAITING -> if (motion > MOVE_T) state = State.MOVING
                State.MOVING -> if (motion < STILL_T) {
                    state = State.STEADYING
                    stillSince = now
                }
                State.STEADYING -> when {
                    motion > STILL_T * 1.5f -> state = State.MOVING
                    now - stillSince >= STILL_MS -> {
                        val shot = lastShot
                        val changed = shot == null || diff(cur, shot) > CHANGE_T
                        if (changed) {
                            lastShot = cur
                            cooldownUntil = now + COOLDOWN_MS
                            onTrigger?.invoke()
                        }
                        state = State.WAITING
                    }
                }
            }
        } finally {
            image.close()
        }
    }

    /** 수동 촬영 직후 호출 → 같은 장면을 자동으로 또 찍지 않게 */
    fun markCaptured() {
        lastShot = prev
    }

    /** Y(밝기) 평면을 GW x GH 로 샘플링하고 평균으로 나눠 노출 변화 영향을 줄인다 */
    private fun thumbnail(image: ImageProxy): FloatArray {
        val plane = image.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val pixStride = plane.pixelStride
        val w = image.width
        val h = image.height
        val out = FloatArray(GW * GH)
        var sum = 0f
        for (gy in 0 until GH) {
            val y = (gy * h + h / 2) / GH
            for (gx in 0 until GW) {
                val x = (gx * w + w / 2) / GW
                val v = (buf.get(y * rowStride + x * pixStride).toInt() and 0xFF).toFloat()
                out[gy * GW + gx] = v
                sum += v
            }
        }
        val mean = (sum / out.size).coerceAtLeast(1f)
        for (i in out.indices) out[i] /= mean
        return out
    }

    /** 평균 절대 차이 (%) */
    private fun diff(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += abs(a[i] - b[i])
        return s / a.size * 100f
    }

    companion object {
        private const val GW = 64
        private const val GH = 48
        // 아래 수치는 실제 기기에서 써보며 조정할 값
        const val MOVE_T = 6f      // 이보다 크면 '페이지 넘기는 중'
        const val STILL_T = 1.5f   // 이보다 작으면 '멈춤'
        const val STILL_MS = 700L  // 멈춘 상태가 이만큼 유지되면 촬영
        const val CHANGE_T = 4f    // 지난 촬영 장면과 이만큼은 달라야 새 페이지로 인정
        const val COOLDOWN_MS = 900L
    }
}
