package com.bookscan.app.ocr

import android.graphics.BitmapFactory
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** OCR 로 읽은 글줄 하나: 글자와 결과 이미지 픽셀 좌표의 상자 */
data class OcrLine(val text: String, val x: Int, val y: Int, val w: Int, val h: Int)

/**
 * 결과 이미지의 글자 읽기(온디바이스 ML Kit, 한국어·영어·숫자). 검색 가능한 PDF 의 보이지 않는 글자층에 쓴다.
 * 결과는 페이지마다 JSON 파일로 저장해 두고 PDF 를 만들 때 읽는다.
 */
object PageOcr {
    private val client: TextRecognizer by lazy {
        TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    }

    /** 긴 변이 이보다 크면 줄여서 읽는다(인식률은 그대로, 속도·메모리 절약). 좌표는 원래 크기로 되돌림 */
    private const val MAX_SIDE = 2600

    /** 백그라운드 스레드에서 호출(동기). 실패하면 빈 목록 */
    fun recognize(image: File): List<OcrLine> {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(image.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return emptyList()
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= MAX_SIDE) sample *= 2
        val bmp = BitmapFactory.decodeFile(image.absolutePath, BitmapFactory.Options().apply { inSampleSize = sample })
            ?: return emptyList()
        try {
            val sx = bounds.outWidth.toFloat() / bmp.width
            val sy = bounds.outHeight.toFloat() / bmp.height
            val result = Tasks.await(client.process(InputImage.fromBitmap(bmp, 0)))
            val lines = ArrayList<OcrLine>()
            for (block in result.textBlocks) for (line in block.lines) {
                val r = line.boundingBox ?: continue
                val t = line.text.trim()
                if (t.isEmpty() || r.width() <= 0 || r.height() <= 0) continue
                lines.add(
                    OcrLine(
                        t,
                        (r.left * sx).toInt(), (r.top * sy).toInt(),
                        maxOf(1, (r.width() * sx).toInt()), maxOf(1, (r.height() * sy).toInt()),
                    )
                )
            }
            return lines
        } finally {
            bmp.recycle()
        }
    }

    fun save(lines: List<OcrLine>, dest: File) {
        val arr = JSONArray()
        for (l in lines) {
            arr.put(JSONObject().put("t", l.text).put("x", l.x).put("y", l.y).put("w", l.w).put("h", l.h))
        }
        dest.writeText(arr.toString())
    }

    fun load(src: File): List<OcrLine> {
        if (!src.exists()) return emptyList()
        return runCatching {
            val arr = JSONArray(src.readText())
            List(arr.length()) { i ->
                val o = arr.getJSONObject(i)
                OcrLine(o.getString("t"), o.getInt("x"), o.getInt("y"), o.getInt("w"), o.getInt("h"))
            }
        }.getOrDefault(emptyList())
    }
}
