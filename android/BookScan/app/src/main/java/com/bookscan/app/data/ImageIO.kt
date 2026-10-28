package com.bookscan.app.data

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max

object ImageIO {

    /** Bitmap(ARGB_8888) → RGB Mat */
    fun bitmapToRgb(bmp: Bitmap): Mat {
        val src = if (bmp.config == Bitmap.Config.ARGB_8888) bmp else bmp.copy(Bitmap.Config.ARGB_8888, false)
        val rgba = Mat()
        Utils.bitmapToMat(src, rgba)
        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()
        if (src !== bmp) src.recycle()
        return rgb
    }

    /** RGB(8UC3) 또는 GRAY(8UC1) Mat → JPEG 파일 */
    fun saveMatJpeg(mat: Mat, file: File, quality: Int = 92) {
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        saveBitmapJpeg(bmp, file, quality)
        bmp.recycle()
    }

    fun saveBitmapJpeg(bmp: Bitmap, file: File, quality: Int = 92) {
        FileOutputStream(file).use { bmp.compress(Bitmap.CompressFormat.JPEG, quality, it) }
    }

    fun loadRgb(file: File): Mat {
        val bmp = BitmapFactory.decodeFile(file.absolutePath, BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }) ?: error("이미지를 읽을 수 없음: ${file.name}")
        val m = bitmapToRgb(bmp)
        bmp.recycle()
        return m
    }

    fun loadMutable(file: File): Bitmap {
        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inMutable = true
        }
        return BitmapFactory.decodeFile(file.absolutePath, opts) ?: error("이미지를 읽을 수 없음: ${file.name}")
    }

    /** 화면 표시용: 긴 변이 maxSide 이하가 되도록 줄여서 읽기 */
    fun decodeSampled(file: File, maxSide: Int): Bitmap? {
        val bounds = imageSize(file) ?: return null
        var sample = 1
        while (max(bounds.first, bounds.second) / (sample * 2) >= maxSide) sample *= 2
        return BitmapFactory.decodeFile(file.absolutePath, BitmapFactory.Options().apply { inSampleSize = sample })
    }

    fun imageSize(file: File): Pair<Int, Int>? {
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, o)
        return if (o.outWidth > 0 && o.outHeight > 0) Pair(o.outWidth, o.outHeight) else null
    }

    /** 파일 이미지를 시계방향 90도 회전해서 덮어쓰기 */
    fun rotateFile90(file: File, quality: Int) {
        val bmp = BitmapFactory.decodeFile(file.absolutePath) ?: return
        val rot = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(90f) }, true)
        saveBitmapJpeg(rot, file, quality)
        bmp.recycle(); rot.recycle()
    }

    /** 사진 앱(Pictures/BookScan)에 JPEG 저장 */
    fun saveToGallery(ctx: Context, src: File, displayName: String): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/BookScan")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val resolver = ctx.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        resolver.openOutputStream(uri)?.use { out -> src.inputStream().use { it.copyTo(out) } }
        values.clear()
        values.put(MediaStore.Images.Media.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
        return uri
    }

    /** 다운로드(Download/BookScan)에 PDF 저장 */
    fun saveToDownloads(ctx: Context, src: File, displayName: String): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, displayName)
            put(MediaStore.Downloads.MIME_TYPE, "application/pdf")
            put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/BookScan")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val resolver = ctx.contentResolver
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return null
        resolver.openOutputStream(uri)?.use { out -> src.inputStream().use { it.copyTo(out) } }
        values.clear()
        values.put(MediaStore.Downloads.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
        return uri
    }
}
