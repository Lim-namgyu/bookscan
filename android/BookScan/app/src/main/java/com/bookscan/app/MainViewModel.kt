package com.bookscan.app

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.bookscan.app.data.ImageIO
import com.bookscan.app.data.ScanPage
import com.bookscan.app.engine.BookScanEngine
import com.bookscan.app.engine.ColorMode
import com.bookscan.app.engine.Layout
import com.bookscan.app.export.PdfExporter
import com.bookscan.app.ocr.PageOcr
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max

class MainViewModel(private val app: Application) : AndroidViewModel(app) {

    // 무거운 보정은 한 번에 하나씩(메모리 보호)
    private val engineThread = Executors.newSingleThreadExecutor().asCoroutineDispatcher()

    @Volatile private var engineInstance: BookScanEngine? = null

    private val engine = viewModelScope.async(engineThread) {
        check(OpenCVLoader.initLocal()) { "OpenCV를 불러오지 못했습니다" }
        val bytes = app.assets.open("uvdoc.onnx").use { it.readBytes() }
        // 책 영역 찾기 모델(없거나 못 읽으면 사진 전체를 책으로 보고 계속)
        val region = runCatching { app.assets.open("u2netp.onnx").use { it.readBytes() } }.getOrNull()
        BookScanEngine(bytes, region).also { engineInstance = it }
    }

    private val sessionDir = File(app.filesDir, "session").apply {
        deleteRecursively(); mkdirs()
    }
    private var nextId = System.currentTimeMillis()

    val pages = mutableStateListOf<ScanPage>()
    var layout by mutableStateOf(Layout.AUTO)
    var colorMode by mutableStateOf(ColorMode.COLOR)
    var removeFingers by mutableStateOf(true)
    var autoCapture by mutableStateOf(false)

    // 촬영음(작은 틱 소리) 켜기/끄기 — 앱을 다시 켜도 유지
    private val prefs = app.getSharedPreferences("bookscan", android.content.Context.MODE_PRIVATE)
    var shutterSound by mutableStateOf(prefs.getBoolean("shutterSound", true))
        private set

    fun setShutterSoundOn(on: Boolean) {
        shutterSound = on
        prefs.edit().putBoolean("shutterSound", on).apply()
    }

    // HDR 촬영(기기 카메라의 다중 프레임 합성: 잡음 적고 그림자·반사 부분도 살아남) — 기기가 지원할 때만 적용
    var hdrCapture by mutableStateOf(prefs.getBoolean("hdrCapture", true))
        private set
    /** 실제로 HDR 이 적용되어 카메라가 열렸는지 (화면 표시용) */
    var hdrActive by mutableStateOf(false)

    fun setHdrCapture(on: Boolean) {
        hdrCapture = on
        prefs.edit().putBoolean("hdrCapture", on).apply()
    }
    var processing by mutableIntStateOf(0)
        private set
    var message by mutableStateOf<String?>(null)

    init {
        // 모델을 미리 올려 첫 촬영 지연을 줄인다
        viewModelScope.launch {
            runCatching { engine.await() }.onFailure { message = "엔진 준비 실패: ${it.message}" }
        }
    }

    /** 촬영된 사진(이미 똑바로 세우고 MAX_SIDE 이하로 줄인 것) 처리. 메인 스레드에서 호출 */
    fun onPhoto(photo: Bitmap) {
        val lay = layout
        val mode = colorMode
        val fingers = removeFingers
        processing++
        viewModelScope.launch {
            try {
                val eng = engine.await()
                val newPages = withContext(engineThread) {
                    // 엔진에 넣는 사진을 그대로 보관 (결과가 이상할 때 원본을 공유해서 확인할 수 있게)
                    val rawFile = File(sessionDir, "${nextId}_raw.jpg")
                    runCatching { ImageIO.saveBitmapJpeg(photo, rawFile, 95) }
                    val rgb = ImageIO.bitmapToRgb(photo)
                    photo.recycle()
                    val dewarped = eng.dewarpPages(rgb, lay)
                    rgb.release()
                    // 흐리게 찍혔으면 알림 (보정은 그대로 하되 다시 찍기를 권함)
                    val sharp = dewarped.minOfOrNull { dp -> eng.sharpness(dp.flat).let { if (it < 0) Double.MAX_VALUE else it } }
                    if (sharp != null && sharp < BookScanEngine.BLUR_WARN) {
                        withContext(Dispatchers.Main) {
                            message = "흐리게 찍혔어요. 폰을 고정하고 책에 조금 더 가까이 대서 다시 찍어 주세요 (초점이 맞으면 초록 테두리)"
                        }
                    }
                    dewarped.map { dp ->
                        val id = nextId++
                        val flatFile = File(sessionDir, "${id}_flat.jpg")
                        ImageIO.saveMatJpeg(dp.flat, flatFile, 95)
                        val result = eng.clean(dp.flat, mode, fingers, dp.skipSide)
                        dp.flat.release()
                        val resultFile = File(sessionDir, "${id}_v0.jpg")
                        ImageIO.saveMatJpeg(result, resultFile, 92)
                        result.release()
                        ScanPage(id, flatFile, resultFile, dp.skipSide, mode, fingers, rawFile = rawFile.takeIf { it.exists() })
                    }
                }
                pages.addAll(newPages)
                newPages.forEach { queueOcr(it.id) }
            } catch (t: Throwable) {
                message = "처리 실패: ${t.message}"
            } finally {
                processing--
            }
        }
    }

    // ---- OCR(검색 가능한 PDF 용 글자층): 페이지가 생기거나 바뀌면 뒤에서 읽어 둔다. 실패해도 스캔에는 영향 없음
    private fun queueOcr(id: Long) {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val cur = pages.firstOrNull { it.id == id } ?: return@launch
                if (cur.ocrFile != null) return@launch
                val lines = PageOcr.recognize(cur.resultFile)
                val f = File(sessionDir, "${cur.id}_v${cur.version}_ocr.json")
                PageOcr.save(lines, f)
                withContext(Dispatchers.Main) {
                    editLock.withLock {
                        val now = pages.firstOrNull { it.id == id } ?: return@withLock
                        if (now.version == cur.version && now.resultFile == cur.resultFile) replace(now.copy(ocrFile = f))
                    }
                }
            }
        }
    }

    /** PDF 만들 때: 아직 안 읽은 페이지는 그 자리에서 읽는다 */
    private fun ocrLines(p: ScanPage): List<com.bookscan.app.ocr.OcrLine> {
        p.ocrFile?.let { return PageOcr.load(it) }
        return runCatching { PageOcr.recognize(p.resultFile) }.getOrDefault(emptyList())
    }

    // 페이지 편집(다시 보정·회전·지우개)은 한 번에 하나씩, 항상 최신 페이지 상태 기준으로
    private val editLock = Mutex()

    private fun edit(id: Long, label: String, block: suspend (ScanPage) -> ScanPage) {
        processing++
        viewModelScope.launch {
            try {
                editLock.withLock {
                    val cur = pages.firstOrNull { it.id == id } ?: return@withLock
                    replace(block(cur).copy(ocrFile = null)) // 결과가 바뀌었으니 글자도 다시 읽는다
                }
                queueOcr(id)
            } catch (t: Throwable) {
                message = "$label 실패: ${t.message}"
            } finally {
                processing--
            }
        }
    }

    /** 색상 모드 / 손가락 제거 여부를 바꿔 다시 보정 */
    fun reprocess(page: ScanPage, mode: ColorMode, fingers: Boolean) = edit(page.id, "다시 보정") { p ->
        val eng = engine.await()
        withContext(engineThread) {
            val flat = ImageIO.loadRgb(p.flatFile)
            val out = eng.clean(flat, mode, fingers, p.skipSide)
            flat.release()
            val v = p.version + 1
            val file = File(sessionDir, "${p.id}_v$v.jpg")
            ImageIO.saveMatJpeg(out, file, 92)
            out.release()
            p.resultFile.delete()
            p.copy(resultFile = file, mode = mode, removeFingers = fingers, version = v)
        }
    }

    fun rotate(page: ScanPage) = edit(page.id, "회전") { p ->
        withContext(Dispatchers.IO) {
            ImageIO.rotateFile90(p.flatFile, 95)
            val v = p.version + 1
            val file = File(sessionDir, "${p.id}_v$v.jpg")
            p.resultFile.copyTo(file, overwrite = true)
            ImageIO.rotateFile90(file, 92)
            p.resultFile.delete()
            // 돌리면 접힌 선 위치가 바뀌므로 손가락 검사 제외 방향은 해제
            p.copy(resultFile = file, version = v, skipSide = null)
        }
    }

    /**
     * 수동 지우개: 결과 이미지 위에 흰색 획을 칠한다.
     * strokes 는 이미지 픽셀 좌표의 점 목록들, brush 는 이미지 픽셀 기준 굵기.
     */
    fun applyEraser(page: ScanPage, strokes: List<List<Pair<Float, Float>>>, brush: Float) {
        if (strokes.isEmpty()) return
        edit(page.id, "지우개") { p ->
            withContext(Dispatchers.IO) {
                val bmp = ImageIO.loadMutable(p.resultFile)
                val paint = Paint().apply {
                    color = android.graphics.Color.WHITE
                    style = Paint.Style.STROKE
                    strokeWidth = brush
                    strokeCap = Paint.Cap.ROUND
                    strokeJoin = Paint.Join.ROUND
                    isAntiAlias = true
                }
                val canvas = Canvas(bmp)
                for (s in strokes) {
                    if (s.isEmpty()) continue
                    val path = Path()
                    path.moveTo(s[0].first, s[0].second)
                    if (s.size == 1) path.lineTo(s[0].first + 0.1f, s[0].second)
                    for (i in 1 until s.size) path.lineTo(s[i].first, s[i].second)
                    canvas.drawPath(path, paint)
                }
                val v = p.version + 1
                val file = File(sessionDir, "${p.id}_v$v.jpg")
                ImageIO.saveBitmapJpeg(bmp, file, 92)
                bmp.recycle()
                p.resultFile.delete()
                p.copy(resultFile = file, version = v)
            }
        }
    }

    fun delete(page: ScanPage) {
        pages.removeAll { it.id == page.id }
        page.flatFile.delete()
        page.resultFile.delete()
        page.ocrFile?.delete()
    }

    fun clearAll() {
        pages.toList().forEach { delete(it) }
    }

    /** PDF 생성 → 다운로드 폴더에 저장하고, 공유용 파일을 돌려준다 */
    suspend fun exportPdf(): File? {
        if (pages.isEmpty()) return null
        val snapshot = pages.toList()
        return withContext(Dispatchers.IO) {
            val name = "BookScan_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".pdf"
            val dir = File(app.cacheDir, "exports").apply { mkdirs() }
            dir.listFiles()?.forEach { it.delete() }
            val dest = File(dir, name)
            val imgs = snapshot.mapNotNull { p ->
                ImageIO.imageSize(p.resultFile)?.let { (w, h) -> PdfExporter.Img(p.resultFile, w, h, ocrLines(p)) }
            }
            PdfExporter.write(imgs, dest)
            ImageIO.saveToDownloads(app, dest, name)
            dest
        }
    }

    /**
     * 확인용 PDF: 엔진에 넣은 원본 사진 + 결과를 그대로(다시 압축하지 않고) 한 PDF에.
     * 결과가 이상할 때 이 파일을 보내면 원본 해상도 그대로 확인할 수 있다(사진으로 보내면 줄어듦).
     */
    suspend fun exportDiagnosticPdf(page: ScanPage): File? {
        val raw = page.rawFile ?: return null
        return withContext(Dispatchers.IO) {
            val name = "BookScan_확인_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".pdf"
            val dir = File(app.cacheDir, "exports").apply { mkdirs() }
            val dest = File(dir, name)
            val imgs = listOf(raw, page.resultFile).mapNotNull { f ->
                ImageIO.imageSize(f)?.let { (w, h) -> PdfExporter.Img(f, w, h) }
            }
            PdfExporter.write(imgs, dest)
            dest
        }
    }

    /** 결과 이미지를 사진 앱(Pictures/BookScan)에 저장 */
    suspend fun saveImagesToGallery(): Int {
        val snapshot = pages.toList()
        return withContext(Dispatchers.IO) {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            snapshot.withIndex().count { (i, p) ->
                ImageIO.saveToGallery(app, p.resultFile, "BookScan_${stamp}_" + String.format(Locale.US, "%03d", i + 1) + ".jpg") != null
            }
        }
    }

    private fun replace(updated: ScanPage) {
        val idx = pages.indexOfFirst { it.id == updated.id }
        if (idx >= 0) pages[idx] = updated
    }

    override fun onCleared() {
        super.onCleared()
        engine.cancel()
        // 보정이 아직 돌고 있을 수 있으므로 엔진 스레드에서 순서대로 닫는다
        engineThread.executor.execute { engineInstance?.let { e -> runCatching { e.close() } } }
        engineThread.close()
    }

    companion object {
        /** 촬영 사진을 엔진에 넣기 전 줄일 크기(긴 변) */
        fun targetScale(w: Int, h: Int): Float =
            (BookScanEngine.MAX_SIDE.toFloat() / max(w, h)).coerceAtMost(1f)
    }
}
