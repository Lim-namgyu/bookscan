package com.bookscan.app.ui

import android.content.Intent
import android.widget.Toast
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.PictureAsPdf
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import com.bookscan.app.MainViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReviewScreen(vm: MainViewModel, onBack: () -> Unit, onOpenPage: (Long) -> Unit) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var working by remember { mutableStateOf(false) }
    var confirmClear by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("스캔한 페이지 ${vm.pages.size}장") },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "카메라로") }
                },
                actions = {
                    IconButton(
                        enabled = vm.pages.isNotEmpty() && !working,
                        onClick = {
                            working = true
                            scope.launch {
                                val n = runCatching { vm.saveImagesToGallery() }.getOrDefault(0)
                                working = false
                                Toast.makeText(ctx, "사진 앱(Pictures/BookScan)에 ${n}장 저장했어요", Toast.LENGTH_SHORT).show()
                            }
                        }
                    ) { Icon(Icons.Filled.SaveAlt, "사진으로 저장") }
                    IconButton(enabled = vm.pages.isNotEmpty(), onClick = { confirmClear = true }) {
                        Icon(Icons.Filled.DeleteSweep, "모두 지우기")
                    }
                },
            )
        },
        bottomBar = {
            Row(
                Modifier.fillMaxWidth().navigationBarsPadding().padding(16.dp),
                horizontalArrangement = Arrangement.Center,
            ) {
                Button(
                    enabled = vm.pages.isNotEmpty() && !working,
                    onClick = {
                        working = true
                        scope.launch {
                            val file = runCatching { vm.exportPdf() }
                                .onFailure { vm.message = "PDF 만들기 실패: ${it.message}" }
                                .getOrNull()
                            working = false
                            if (file != null) {
                                Toast.makeText(ctx, "다운로드/BookScan 폴더에 저장했어요", Toast.LENGTH_SHORT).show()
                                val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".fileprovider", file)
                                val send = Intent(Intent.ACTION_SEND).apply {
                                    type = "application/pdf"
                                    putExtra(Intent.EXTRA_STREAM, uri)
                                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                }
                                ctx.startActivity(Intent.createChooser(send, "PDF 공유"))
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                ) {
                    if (working) {
                        CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                    } else {
                        Icon(Icons.Filled.PictureAsPdf, null)
                        Spacer(Modifier.width(8.dp))
                        Text("PDF로 저장·공유")
                    }
                }
            }
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            if (vm.processing > 0) LinearProgressIndicator(Modifier.fillMaxWidth())
            if (vm.pages.isEmpty()) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("아직 찍은 페이지가 없어요", color = Color.Gray)
                }
            } else {
                LazyVerticalGrid(
                    columns = GridCells.Fixed(3),
                    modifier = Modifier.fillMaxSize().padding(horizontal = 8.dp),
                ) {
                    itemsIndexed(vm.pages, key = { _, p -> p.id }) { i, p ->
                        Column(Modifier.padding(6.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                            PageImage(
                                p.resultFile, p.version, 480,
                                Modifier.fillMaxWidth().aspectRatio(0.72f).clip(RoundedCornerShape(6.dp))
                                    .clickable { onOpenPage(p.id) },
                                ContentScale.Fit,
                            )
                            Text("${i + 1}", fontSize = 12.sp, color = Color.LightGray, modifier = Modifier.padding(top = 2.dp))
                        }
                    }
                }
            }
        }
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("모두 지울까요?") },
            text = { Text("찍은 페이지 ${vm.pages.size}장을 모두 지웁니다. 저장하지 않은 페이지는 되살릴 수 없어요.") },
            confirmButton = {
                TextButton(onClick = { vm.clearAll(); confirmClear = false }) { Text("지우기") }
            },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("취소") } },
        )
    }
}
