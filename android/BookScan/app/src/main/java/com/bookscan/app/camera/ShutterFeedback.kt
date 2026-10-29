package com.bookscan.app.camera

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin

/**
 * 촬영 피드백: 짧은 진동 + 아주 작은 '틱' 소리.
 * 시스템 셔터음(MediaActionSound)은 기기 설정에 따라 크게 나고 줄일 수 없어서, 직접 만든 짧은 소리를
 * 미디어 음량 기준으로 작게 낸다(미디어 음량을 줄이면 같이 작아짐). 소리를 끄면 진동만 남는다.
 */
class ShutterFeedback(ctx: Context) {
    @Volatile var soundOn = true

    private val vibrator: Vibrator? = runCatching {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (ctx.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            ctx.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
    }.getOrNull()

    private val track: AudioTrack? = runCatching { buildTrack() }.getOrNull()

    private fun buildTrack(): AudioTrack {
        val rate = 44100
        val n = rate * 60 / 1000 // 60ms
        val pcm = ShortArray(n)
        for (i in 0 until n) {
            val t = i.toDouble() / rate
            val env = exp(-t * 80.0) // 빠르게 사그라드는 '틱'
            val v = (sin(2 * PI * 1800.0 * t) * 0.7 + sin(2 * PI * 3600.0 * t) * 0.3) * env * GAIN
            pcm[i] = (v * Short.MAX_VALUE).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
        }
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()
        val fmt = AudioFormat.Builder()
            .setSampleRate(rate)
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        val tr = AudioTrack(attrs, fmt, n * 2, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE)
        tr.write(pcm, 0, n)
        tr.setVolume(VOLUME)
        return tr
    }

    fun play() {
        runCatching { vibrator?.vibrate(VibrationEffect.createPredefined(VibrationEffect.EFFECT_CLICK)) }
        if (!soundOn) return
        val tr = track ?: return
        runCatching {
            if (tr.playState != AudioTrack.PLAYSTATE_STOPPED) tr.stop()
            tr.reloadStaticData()
            tr.play()
        }
    }

    fun release() {
        runCatching { track?.release() }
    }

    companion object {
        private const val GAIN = 0.22   // 파형 크기(0~1)
        private const val VOLUME = 0.5f // 트랙 음량(미디어 음량에 곱해짐)
    }
}
