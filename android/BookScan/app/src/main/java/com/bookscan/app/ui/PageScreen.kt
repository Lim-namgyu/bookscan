package com.bookscan.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bookscan.app.MainViewModel
import com.bookscan.app.data.ImageIO
import com.bookscan.app.data.ScanPage
import com.bookscan.app.engine.ColorMode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.min

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PageScreen(vm: MainViewModel, page: ScanPage, onBack: () -> Unit) {
    val index = vm.pages.indexOfFirst { it.id == page.id }
    var eraser by remember { mutableStateOf(false) }
    val strokes = remember { mutableStateListOf<List<Offset>>() }
    var current by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var boxSize by remember { mutableStateOf(IntSize.Zero) }
    val brushPx = with(LocalDensity.current) { 26.dp.toPx() }
    val imgSize by produceState<Pair<Int, Int>?>(null, page.resultFile, page.version) {
        value = withContext(Dispatchers.IO) { ImageIO.imageSize(page.resultFile) }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("${index + 1} / ${vm.pages.size}") },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "목록") } },
                actions = {
                    IconButton(onClick = { vm.delete(page); onBack() }) { Icon(Icons.Filled.Delete, "이 페이지 삭제") }
                },
            )
        },
        bottomBar = {
            Column(Modifier.fillMaxWidth().navigationBarsPadding().padding(12.dp)) {
                if (eraser) {
                    Text("지울 곳을 손가락으로 문지르세요", fontSize = 13.sp, color = Color.LightGray)
                    Row(
                        Modifier.fillMaxWidth().padding(top = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        TextButton(onClick = { strokes.clear(); eraser = false }) { Text("취소") }
                        IconButton(enabled = strokes.isNotEmpty(), onClick = { strokes.removeAt(strokes.lastIndex) }) {
                            Icon(Icons.AutoMirrored.Filled.Undo, "되돌리기")
                        }
                        Box(Modifier.weight(1f))
                        Button(
                            enabled = strokes.isNotEmpty() && imgSize != null && boxSize != IntSize.Zero,
                            onClick = {
                                val (iw, ih) = imgSize!!
                                val scale = min(boxSize.width / iw.toFloat(), boxSize.height / ih.toFloat())
                                val ox = (boxSize.width - iw * scale) / 2f
                                val oy = (boxSize.height - ih * scale) / 2f
                                val mapped = strokes.map { s -> s.map { o -> Pair((o.x - ox) / scale, (o.y - oy) / scale) } }
                                vm.applyEraser(page, mapped, brushPx / scale)
                                strokes.clear()
                                eraser = false
                            },
                        ) { Text("적용") }
                    }
                } else {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        for ((label, m) in listOf("컬러" to ColorMode.COLOR, "회색" to ColorMode.GRAY, "흑백" to ColorMode.BW)) {
                            ModeChip(label, page.mode == m) {
                                if (page.mode != m) vm.reprocess(page, m, page.removeFingers)
                            }
                        }
                        ModeChip("손가락 제거", page.removeFingers) { vm.reprocess(page, page.mode, !page.removeFingers) }
                    }
                    Row(
                        Modifier.fillMaxWidth().padding(top = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        OutlinedButton(onClick = { vm.rotate(page) }, modifier = Modifier.weight(1f)) {
                            Icon(Icons.Filled.Refresh, null)
                            Text(" 회전")
                        }
                        OutlinedButton(onClick = { eraser = true }, modifier = Modifier.weight(1f)) {
                            Icon(Icons.Filled.AutoFixHigh, null)
                            Text(" 지우개")
                        }
                    }
                }
            }
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            if (vm.processing > 0) LinearProgressIndicator(Modifier.fillMaxWidth())
            Box(
                Modifier.fillMaxSize().background(Color(0xFF0C0C0C)).onSizeChanged { boxSize = it }
            ) {
                PageImage(page.resultFile, page.version, 2048, Modifier.fillMaxSize())
                if (eraser) {
                    Canvas(
                        Modifier.fillMaxSize().pointerInput(Unit) {
                            detectDragGestures(
                                onDragStart = { current = listOf(it) },
                                onDrag = { change, _ -> current = current + change.position },
                                onDragEnd = {
                                    if (current.isNotEmpty()) strokes.add(current)
                                    current = emptyList()
                                },
                                onDragCancel = { current = emptyList() },
                            )
                        }
                    ) {
                        val all = strokes + listOf(current)
                        for (s in all) {
                            if (s.isEmpty()) continue
                            val path = Path().apply {
                                moveTo(s[0].x, s[0].y)
                                for (i in 1 until s.size) lineTo(s[i].x, s[i].y)
                            }
                            drawPath(
                                path, Color(0x88FF5252),
                                style = Stroke(width = brushPx, cap = StrokeCap.Round, join = StrokeJoin.Round)
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ModeChip(label: String, selected: Boolean, onClick: () -> Unit) {
    FilterChip(
        selected = selected,
        onClick = onClick,
        label = { Text(label, fontSize = 13.sp) },
        modifier = Modifier.padding(end = 6.dp),
    )
}
