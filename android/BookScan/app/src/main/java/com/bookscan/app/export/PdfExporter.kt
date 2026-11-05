package com.bookscan.app.export

import com.bookscan.app.ocr.OcrLine
import java.io.File
import java.io.FilterOutputStream
import java.io.OutputStream

/**
 * JPEG 를 다시 압축하지 않고 그대로 넣는 최소 PDF 작성기.
 * (Android PdfDocument 는 이미지를 무압축에 가깝게 넣어 파일이 매우 커짐)
 * 페이지 폭은 A4 폭(595pt)으로 맞추고 높이는 이미지 비율대로.
 *
 * OCR 글줄이 있으면 이미지 위에 보이지 않는 글자층(렌더 모드 3)을 같은 자리에 얹는다 → PDF 뷰어에서 검색·복사·선택이 됨.
 * 글꼴은 글리프 없는 가짜 CID 글꼴(내장하지 않음, Identity-H: 글자 코드 = 유니코드)에 ToUnicode 맵을 붙인다.
 * 보이지 않으므로 모양은 상관없고, 어떤 뷰어든 ToUnicode 로 유니코드를 바로 풀 수 있어 검색·복사가 된다
 * (tesseract/ocrmypdf 가 쓰는 방식). 글자 폭: 한글·한자 1em, 영문·숫자 0.5em, 공백 0.3em.
 */
object PdfExporter {

    class Img(val jpeg: File, val width: Int, val height: Int, val lines: List<OcrLine> = emptyList())

    private const val FONT_OBJ = 3      // Type0 글꼴
    private const val CID_OBJ = 4       // CIDFont
    private const val DESC_OBJ = 5      // FontDescriptor
    private const val TOUNI_OBJ = 6     // ToUnicode CMap
    private const val FIRST_PAGE_OBJ = 7

    /** 글자 폭(1/1000 em): /W 배열과 같은 값이어야 자리가 맞는다 */
    private fun charWidth(c: Int): Int = when {
        c == 32 -> 300
        c in 33..126 -> 500
        else -> 1000
    }

    private class Counting(out: OutputStream) : FilterOutputStream(out) {
        var count = 0L
        override fun write(b: Int) {
            out.write(b); count++
        }
        override fun write(b: ByteArray, off: Int, len: Int) {
            out.write(b, off, len); count += len
        }
    }

    fun write(images: List<Img>, dest: File) {
        require(images.isNotEmpty()) { "페이지가 없습니다" }
        val n = images.size
        val objCount = FIRST_PAGE_OBJ - 1 + 3 * n
        val offsets = LongArray(objCount + 1)
        Counting(dest.outputStream().buffered(1 shl 16)).use { out ->
            fun w(s: String) = out.write(s.toByteArray(Charsets.ISO_8859_1))

            w("%PDF-1.4\n")
            out.write(byteArrayOf('%'.code.toByte(), 0xE2.toByte(), 0xE3.toByte(), 0xCF.toByte(), 0xD3.toByte(), '\n'.code.toByte()))

            offsets[1] = out.count
            w("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

            offsets[2] = out.count
            val kids = (0 until n).joinToString(" ") { "${FIRST_PAGE_OBJ + 3 * it} 0 R" }
            w("2 0 obj\n<< /Type /Pages /Kids [$kids] /Count $n >>\nendobj\n")

            // 보이지 않는 글자층용 글꼴(글리프 없음, 내장하지 않음) + ToUnicode
            offsets[FONT_OBJ] = out.count
            w(
                "$FONT_OBJ 0 obj\n<< /Type /Font /Subtype /Type0 /BaseFont /BookScanOCR /Encoding /Identity-H " +
                    "/DescendantFonts [$CID_OBJ 0 R] /ToUnicode $TOUNI_OBJ 0 R >>\nendobj\n"
            )
            offsets[CID_OBJ] = out.count
            w(
                "$CID_OBJ 0 obj\n<< /Type /Font /Subtype /CIDFontType2 /BaseFont /BookScanOCR " +
                    "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> " +
                    "/FontDescriptor $DESC_OBJ 0 R /DW 1000 /W [32 [300] 33 126 500] /CIDToGIDMap /Identity >>\nendobj\n"
            )
            offsets[DESC_OBJ] = out.count
            w(
                "$DESC_OBJ 0 obj\n<< /Type /FontDescriptor /FontName /BookScanOCR /Flags 4 " +
                    "/FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 900 /Descent -200 /CapHeight 700 /StemV 80 >>\nendobj\n"
            )
            // ToUnicode: 2바이트 코드 = 유니코드(BMP). 256개 범위로 나눠 적는다(한 범위의 끝 바이트가 넘치지 않게)
            val cmap = StringBuilder()
            cmap.append("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n")
            cmap.append("/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n")
            cmap.append("/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n")
            cmap.append("1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n")
            var hi = 0
            while (hi < 256) {
                val cnt = minOf(100, 256 - hi)
                cmap.append("$cnt beginbfrange\n")
                for (k in hi until hi + cnt) {
                    val p = String.format(java.util.Locale.US, "%02X", k)
                    cmap.append("<${p}00> <${p}FF> <${p}00>\n")
                }
                cmap.append("endbfrange\n")
                hi += cnt
            }
            cmap.append("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n")
            val cmapBytes = cmap.toString().toByteArray(Charsets.ISO_8859_1)
            offsets[TOUNI_OBJ] = out.count
            w("$TOUNI_OBJ 0 obj\n<< /Length ${cmapBytes.size} >>\nstream\n")
            out.write(cmapBytes)
            w("\nendstream\nendobj\n")

            images.forEachIndexed { i, img ->
                val pageObj = FIRST_PAGE_OBJ + 3 * i
                val contentObj = pageObj + 1
                val imageObj = pageObj + 2
                val pw = 595.0
                val ph = pw * img.height / img.width
                val pws = fmt(pw)
                val phs = fmt(ph)

                offsets[pageObj] = out.count
                w(
                    "$pageObj 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 $pws $phs] " +
                        "/Resources << /XObject << /Im0 $imageObj 0 R >> /Font << /F1 $FONT_OBJ 0 R >> >> " +
                        "/Contents $contentObj 0 R >>\nendobj\n"
                )

                val content = ("q\n$pws 0 0 $phs 0 0 cm\n/Im0 Do\nQ\n" + textLayer(img, pw / img.width, ph))
                    .toByteArray(Charsets.ISO_8859_1)
                offsets[contentObj] = out.count
                w("$contentObj 0 obj\n<< /Length ${content.size} >>\nstream\n")
                out.write(content)
                w("endstream\nendobj\n")

                val len = img.jpeg.length()
                offsets[imageObj] = out.count
                w(
                    "$imageObj 0 obj\n<< /Type /XObject /Subtype /Image /Width ${img.width} /Height ${img.height} " +
                        "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length $len >>\nstream\n"
                )
                img.jpeg.inputStream().use { it.copyTo(out) }
                w("\nendstream\nendobj\n")
            }

            val xref = out.count
            val sb = StringBuilder()
            sb.append("xref\n0 ${objCount + 1}\n")
            sb.append("0000000000 65535 f \n")
            for (k in 1..objCount) sb.append(String.format(java.util.Locale.US, "%010d 00000 n \n", offsets[k]))
            sb.append("trailer\n<< /Size ${objCount + 1} /Root 1 0 R >>\nstartxref\n$xref\n%%EOF\n")
            w(sb.toString())
        }
    }

    /**
     * 보이지 않는 글자층: 글줄마다 상자 높이를 글꼴 크기로, 상자 폭에 맞게 가로 배율(Tz)을 조절해 같은 자리에 놓는다.
     * scale: 이미지 픽셀 → pt, ph: 페이지 높이(pt). 글자는 2바이트 유니코드 16진 문자열(Identity-H).
     */
    private fun textLayer(img: Img, scale: Double, ph: Double): String {
        if (img.lines.isEmpty()) return ""
        val sb = StringBuilder("BT\n3 Tr\n")
        for (l in img.lines) {
            val hex = StringBuilder()
            var natural = 0 // 글자 폭 합(1/1000 em)
            for (ch in l.text) {
                val c = ch.code
                if (c in 0xD800..0xDFFF) continue // 보조 평면 문자는 2바이트로 못 적음 → 건너뜀
                hex.append(String.format(java.util.Locale.US, "%04X", c))
                natural += charWidth(c)
            }
            if (natural == 0) continue
            val fs = (l.h * scale).coerceAtLeast(1.0)
            val width = l.w * scale
            // 글자 폭 합을 상자 폭에 맞춤 (가로 배율 20~400%)
            val tz = (width / (natural / 1000.0 * fs) * 100.0).coerceIn(20.0, 400.0)
            val x = l.x * scale
            val y = ph - (l.y + l.h * 0.85) * scale // 기준선 ≈ 상자 아래에서 15%
            sb.append("/F1 ${fmt(fs)} Tf ${fmt(tz)} Tz 1 0 0 1 ${fmt(x)} ${fmt(y)} Tm <").append(hex).append("> Tj\n")
        }
        sb.append("ET\n")
        return sb.toString()
    }

    private fun fmt(v: Double) = String.format(java.util.Locale.US, "%.2f", v)
}
