package com.bookscan.app.export

import java.io.File
import java.io.FilterOutputStream
import java.io.OutputStream

/**
 * JPEG 를 다시 압축하지 않고 그대로 넣는 최소 PDF 작성기.
 * (Android PdfDocument 는 이미지를 무압축에 가깝게 넣어 파일이 매우 커짐)
 * 페이지 폭은 A4 폭(595pt)으로 맞추고 높이는 이미지 비율대로.
 */
object PdfExporter {

    class Img(val jpeg: File, val width: Int, val height: Int)

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
        val objCount = 2 + 3 * n
        val offsets = LongArray(objCount + 1)
        Counting(dest.outputStream().buffered(1 shl 16)).use { out ->
            fun w(s: String) = out.write(s.toByteArray(Charsets.ISO_8859_1))

            w("%PDF-1.4\n")
            out.write(byteArrayOf('%'.code.toByte(), 0xE2.toByte(), 0xE3.toByte(), 0xCF.toByte(), 0xD3.toByte(), '\n'.code.toByte()))

            offsets[1] = out.count
            w("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

            offsets[2] = out.count
            val kids = (0 until n).joinToString(" ") { "${3 + 3 * it} 0 R" }
            w("2 0 obj\n<< /Type /Pages /Kids [$kids] /Count $n >>\nendobj\n")

            images.forEachIndexed { i, img ->
                val pageObj = 3 + 3 * i
                val contentObj = pageObj + 1
                val imageObj = pageObj + 2
                val pw = 595.0
                val ph = pw * img.height / img.width
                val pws = fmt(pw)
                val phs = fmt(ph)

                offsets[pageObj] = out.count
                w(
                    "$pageObj 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 $pws $phs] " +
                        "/Resources << /XObject << /Im0 $imageObj 0 R >> >> /Contents $contentObj 0 R >>\nendobj\n"
                )

                val content = "q\n$pws 0 0 $phs 0 0 cm\n/Im0 Do\nQ\n".toByteArray(Charsets.ISO_8859_1)
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

    private fun fmt(v: Double) = String.format(java.util.Locale.US, "%.2f", v)
}
