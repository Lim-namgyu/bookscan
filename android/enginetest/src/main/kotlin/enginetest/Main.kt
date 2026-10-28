package enginetest

import com.bookscan.app.engine.BookScanEngine
import com.bookscan.app.engine.ColorMode
import com.bookscan.app.engine.Layout
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.max

/**
 * 사용: enginetest <assets 폴더> <출력 폴더> <사진…>
 * 앱의 onPhoto 와 같은 순서(긴 변 MAX_SIDE 이하로 줄임 → dewarpPages → clean)로 처리해서
 * <출력>/<이름>_p<n>.jpg 로 저장한다. 파이썬 bookscan.py 의 출력 이름과 같다.
 */
fun main(args: Array<String>) {
    nu.pattern.OpenCV.loadLocally()
    val assets = File(args[0])
    val out = File(args[1]).also { it.mkdirs() }
    val layout = Layout.valueOf((System.getenv("LAYOUT") ?: "AUTO").uppercase())
    val mode = ColorMode.valueOf((System.getenv("MODE") ?: "COLOR").uppercase())
    val engine = BookScanEngine(
        File(assets, "uvdoc.onnx").readBytes(),
        File(assets, "u2netp.onnx").takeIf { it.exists() }?.readBytes()
    )
    var failed = 0
    for (path in args.drop(2)) {
        val name = File(path).nameWithoutExtension
        val bgr = Imgcodecs.imread(path)
        if (bgr.empty()) {
            println("$name: 읽기 실패"); failed++; continue
        }
        val rgb = Mat()
        Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB)
        bgr.release()
        val s = (BookScanEngine.MAX_SIDE.toDouble() / max(rgb.cols(), rgb.rows())).coerceAtMost(1.0)
        if (s < 1.0) Imgproc.resize(rgb, rgb, Size(), s, s, Imgproc.INTER_AREA)
        val t0 = System.nanoTime()
        try {
            val pages = engine.dewarpPages(rgb, layout)
            val t1 = System.nanoTime()
            val sizes = ArrayList<String>()
            pages.forEachIndexed { i, dp ->
                val res = engine.clean(dp.flat, mode, true, dp.skipSide)
                dp.flat.release()
                val save = if (res.channels() == 3) Mat().also { Imgproc.cvtColor(res, it, Imgproc.COLOR_RGB2BGR) } else res
                Imgcodecs.imwrite(File(out, "${name}_p${i + 1}.jpg").path, save, MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 92))
                sizes.add("${res.cols()}x${res.rows()}")
                if (save !== res) save.release()
                res.release()
            }
            val t2 = System.nanoTime()
            println("%s: %d pages %s  dewarp %.1fs clean %.1fs".format(name, pages.size, sizes, (t1 - t0) / 1e9, (t2 - t1) / 1e9))
        } catch (t: Throwable) {
            println("$name: 실패 $t"); t.printStackTrace(); failed++
        } finally {
            rgb.release()
        }
    }
    engine.close()
    if (failed > 0) {
        System.err.println("$failed 장 실패")
        System.exit(1)
    }
}
