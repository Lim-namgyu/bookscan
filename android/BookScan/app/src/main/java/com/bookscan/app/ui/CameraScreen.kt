package com.bookscan.app.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.MediaActionSound
import android.view.OrientationEventListener
import android.view.Surface
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Autorenew
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.bookscan.app.MainViewModel
import com.bookscan.app.camera.CaptureReadiness
import com.bookscan.app.engine.BookRegion
import com.bookscan.app.engine.ColorMode
import com.bookscan.app.engine.Layout
import kotlinx.coroutines.delay
import org.opencv.android.OpenCVLoader
import java.util.concurrent.Executors
import kotlin.math.atan2
import kotlin.math.sqrt

@Composable
fun CameraScreen(vm: MainViewModel, onOpenReview: () -> Unit) {
    val ctx = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) launcher.launch(Manifest.permission.CAMERA) }

    if (!granted) {
        Column(
            Modifier.fillMaxSize().background(Color.Black).padding(32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("책을 찍으려면 카메라 권한이 필요해요", color = Color.White, textAlign = TextAlign.Center)
            Spacer(Modifier.height(16.dp))
            Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) { Text("권한 허용") }
        }
        return
    }
    CameraContent(vm, onOpenReview)
}

@Composable
private fun CameraContent(vm: MainViewModel, onOpenReview: () -> Unit) {
    val ctx = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainExecutor = remember { ContextCompat.getMainExecutor(ctx) }
    val previewView = remember { PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FIT_CENTER } }
    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    // 4000x3000 근처로 제한(초고화소 센서에서 메모리 부족 방지, 엔진은 3000px로 처리)
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            android.util.Size(4000, 3000),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER,
                        )
                    )
                    .build()
            )
            .build()
    }
    val readiness = remember {
        CaptureReadiness {
            // 책 영역 모델(2.3MB)을 프리뷰 분석 스레드에서 올린다. 실패하면 책 위치 안내 없이 나머지만 판단
            if (!OpenCVLoader.initLocal()) null
            else BookRegion(ctx.assets.open("u2netp.onnx").use { it.readBytes() })
        }
    }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }
    val sound = remember { MediaActionSound().apply { load(MediaActionSound.SHUTTER_CLICK) } }

    var camera by remember { mutableStateOf<Camera?>(null) }
    var torchOn by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var flash by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf<CaptureReadiness.Status?>(null) }

    val capture: () -> Unit = {
        if (!busy) {
            busy = true
            flash = true
            sound.play(MediaActionSound.SHUTTER_CLICK)
            imageCapture.takePicture(captureExecutor, object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        val rot = image.imageInfo.rotationDegrees
                        val raw = image.toBitmap()
                        image.close()
                        // 똑바로 세우면서 처리 해상도(긴 변 3000px)로 줄이기
                        val sc = MainViewModel.targetScale(raw.width, raw.height)
                        val m = Matrix().apply {
                            postScale(sc, sc)
                            postRotate(rot.toFloat())
                        }
                        val upright = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                        if (upright !== raw) raw.recycle()
                        readiness.markCaptured()
                        mainExecutor.execute {
                            busy = false
                            vm.onPhoto(upright)
                        }
                    } catch (t: Throwable) {
                        runCatching { image.close() }
                        mainExecutor.execute {
                            busy = false
                            vm.message = "사진 처리 실패: ${t.message}"
                        }
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    mainExecutor.execute {
                        busy = false
                        vm.message = "촬영 실패: ${exception.message}"
                    }
                }
            })
        }
    }

    // 촬영 준비 상태 · 자동 촬영 연결
    SideEffect {
        readiness.onTrigger = { mainExecutor.execute { capture() } }
        readiness.onStatus = { st -> mainExecutor.execute { status = st } }
    }
    LaunchedEffect(vm.autoCapture) { readiness.autoEnabled = vm.autoCapture }
    LaunchedEffect(torchOn, camera) { camera?.cameraControl?.enableTorch(torchOn) }

    // 폰 기울기(책과 나란한지): 중력 방향과 화면 수직축 사이 각도
    DisposableEffect(Unit) {
        val sm = ctx.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val sensor = sm.getDefaultSensor(Sensor.TYPE_GRAVITY) ?: sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val listener = object : SensorEventListener {
            var fx = 0f; var fy = 0f; var fz = 9.8f
            override fun onSensorChanged(e: SensorEvent) {
                fx = fx * 0.8f + e.values[0] * 0.2f
                fy = fy * 0.8f + e.values[1] * 0.2f
                fz = fz * 0.8f + e.values[2] * 0.2f
                val tilt = Math.toDegrees(atan2(sqrt(fx * fx + fy * fy).toDouble(), Math.abs(fz).toDouble())).toFloat()
                readiness.tiltDeg = tilt
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        if (sensor != null) sm.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_UI)
        onDispose { sm.unregisterListener(listener) }
    }

    // 화면은 세로로 고정이지만, 폰을 가로로 들고 찍으면 사진도 가로로 저장되게(양면 촬영 시 흔함)
    DisposableEffect(Unit) {
        val listener = object : OrientationEventListener(ctx) {
            var current = Surface.ROTATION_0
            override fun onOrientationChanged(orientation: Int) {
                // 폰을 거의 수평으로 들면(책 위에서 내려찍기) 방향을 알 수 없음 → 마지막 방향 유지
                if (orientation == ORIENTATION_UNKNOWN) return
                // 경계(45°)에서 왔다 갔다 하지 않도록, 현재 방향의 중심에서 60° 넘게 벗어날 때만 바꿈
                val center = when (current) {
                    Surface.ROTATION_270 -> 90
                    Surface.ROTATION_180 -> 180
                    Surface.ROTATION_90 -> 270
                    else -> 0
                }
                val diff = Math.abs(((orientation - center) % 360 + 540) % 360 - 180)
                if (diff <= 60) return
                current = when (orientation) {
                    in 45 until 135 -> Surface.ROTATION_270
                    in 135 until 225 -> Surface.ROTATION_180
                    in 225 until 315 -> Surface.ROTATION_90
                    else -> Surface.ROTATION_0
                }
                imageCapture.targetRotation = current
            }
        }
        listener.enable()
        onDispose { listener.disable() }
    }
    LaunchedEffect(flash) {
        if (flash) {
            delay(120); flash = false
        }
    }

    DisposableEffect(lifecycleOwner) {
        val future = ProcessCameraProvider.getInstance(ctx)
        val analysisExecutor = Executors.newSingleThreadExecutor()
        future.addListener({
            val provider = future.get()
            val ratio = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()
            val preview = Preview.Builder().setResolutionSelector(ratio).build()
            preview.setSurfaceProvider(previewView.surfaceProvider)
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(ratio)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor, readiness)
            try {
                provider.unbindAll()
                camera = provider.bindToLifecycle(
                    lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture, analysis
                )
            } catch (e: Exception) {
                vm.message = "카메라를 시작하지 못했어요: ${e.message}"
            }
        }, mainExecutor)
        onDispose {
            // 카메라를 먼저 해제한 뒤 실행기를 닫아야, 진행 중이던 촬영 콜백이 닫힌 실행기로 가지 않는다
            runCatching { future.get().unbindAll() }
            analysisExecutor.execute { readiness.close() }
            analysisExecutor.shutdown()
            captureExecutor.shutdown()
            sound.release()
        }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        // 탭해서 초점 맞추기
        Box(
            Modifier.fillMaxSize().pointerInput(Unit) {
                detectTapGestures { pos ->
                    val point = previewView.meteringPointFactory.createPoint(pos.x, pos.y)
                    camera?.cameraControl?.startFocusAndMetering(FocusMeteringAction.Builder(point).build())
                }
            }
        )

        // 책 영역 테두리: 준비되면 초록, 아니면 흰색
        val st = status
        val ready = st?.ready == true
        val readyColor = Color(0xFF4CD964)
        if (st?.box != null) {
            val box = st.box
            Canvas(Modifier.fillMaxSize()) {
                // 프리뷰(FIT_CENTER, 세로 화면에서 3:4)가 실제로 그려진 영역
                val aspect = 3f / 4f
                val cw = minOf(size.width, size.height * aspect)
                val ch = cw / aspect
                val ox = (size.width - cw) / 2f
                val oy = (size.height - ch) / 2f
                val l = ox + box.left * cw
                val t = oy + box.top * ch
                val r = ox + box.right * cw
                val b = oy + box.bottom * ch
                drawRoundRect(
                    color = if (ready) readyColor else Color.White.copy(alpha = 0.8f),
                    topLeft = Offset(l, t), size = Size(r - l, b - t),
                    cornerRadius = CornerRadius(16f, 16f),
                    style = Stroke(width = if (ready) 6f else 3f),
                )
            }
        }

        if (flash) Box(Modifier.fillMaxSize().background(Color.White.copy(alpha = 0.35f)))

        // ---- 위쪽: 촬영 옵션
        Column(
            Modifier.align(Alignment.TopCenter).fillMaxWidth().statusBarsPadding()
                .background(Color.Black.copy(alpha = 0.45f)).padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                OptionChip("자동", vm.layout == Layout.AUTO) { vm.layout = Layout.AUTO }
                OptionChip("한 쪽", vm.layout == Layout.SINGLE) { vm.layout = Layout.SINGLE }
                OptionChip("양면", vm.layout == Layout.DOUBLE) { vm.layout = Layout.DOUBLE }
                Spacer(Modifier.weight(1f))
                IconButton(onClick = { torchOn = !torchOn }) {
                    Icon(if (torchOn) Icons.Filled.FlashOn else Icons.Filled.FlashOff, "조명", tint = Color.White)
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                OptionChip("컬러", vm.colorMode == ColorMode.COLOR) { vm.colorMode = ColorMode.COLOR }
                OptionChip("회색", vm.colorMode == ColorMode.GRAY) { vm.colorMode = ColorMode.GRAY }
                OptionChip("흑백", vm.colorMode == ColorMode.BW) { vm.colorMode = ColorMode.BW }
                Spacer(Modifier.width(8.dp))
                OptionChip("손가락 제거", vm.removeFingers) { vm.removeFingers = !vm.removeFingers }
            }
        }

        // ---- 촬영 준비 상태 안내
        Box(
            Modifier.align(Alignment.TopCenter).statusBarsPadding().padding(top = 118.dp, start = 24.dp, end = 24.dp),
            contentAlignment = Alignment.TopCenter,
        ) {
            Row(
                Modifier.clip(RoundedCornerShape(20.dp))
                    .background(if (ready) readyColor.copy(alpha = 0.9f) else Color.Black.copy(alpha = 0.6f))
                    .padding(horizontal = 14.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(
                    Modifier.size(10.dp).clip(CircleShape).background(
                        when {
                            ready -> Color.White
                            st == null || st.issue == CaptureReadiness.Issue.BLURRY || st.issue == CaptureReadiness.Issue.MOVING -> Color(0xFFFFC107)
                            else -> Color(0xFFFF5252)
                        }
                    )
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    statusText(st, vm.autoCapture),
                    color = if (ready) Color.Black else Color.White,
                    fontSize = 14.sp, fontWeight = FontWeight.Medium,
                )
            }
        }

        // ---- 아래쪽: 미리보기 · 셔터 · 자동촬영
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().navigationBarsPadding()
                .background(Color.Black.copy(alpha = 0.45f)).padding(vertical = 14.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (vm.processing > 0) {
                Text("보정 중 ${vm.processing}장…", color = Color.White, fontSize = 13.sp)
                Spacer(Modifier.height(8.dp))
            }
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 28.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // 최근 페이지 썸네일
                Box(
                    Modifier.size(58.dp).clip(RoundedCornerShape(8.dp)).background(Color(0xFF333333))
                        .clickable { onOpenReview() },
                    contentAlignment = Alignment.Center,
                ) {
                    val last = vm.pages.lastOrNull()
                    if (last != null) {
                        PageImage(last.resultFile, last.version, 200, Modifier.fillMaxSize(), ContentScale.Crop)
                        Box(
                            Modifier.align(Alignment.TopEnd).padding(3.dp).clip(CircleShape)
                                .background(MaterialTheme.colorScheme.primary).padding(horizontal = 6.dp, vertical = 1.dp)
                        ) {
                            Text("${vm.pages.size}", color = Color.Black, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                }

                // 셔터 (준비되면 초록 테두리)
                Box(
                    Modifier.size(78.dp).clip(CircleShape).border(4.dp, if (ready) readyColor else Color.White, CircleShape)
                        .padding(7.dp).clip(CircleShape)
                        .background(if (busy) Color.Gray else if (ready) readyColor else Color.White)
                        .clickable(enabled = !busy) { capture() }
                )

                // 자동 촬영
                Column(
                    Modifier.width(58.dp).clickable { vm.autoCapture = !vm.autoCapture },
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    val on = vm.autoCapture
                    Icon(
                        Icons.Filled.Autorenew, "자동 촬영",
                        tint = if (on) MaterialTheme.colorScheme.primary else Color.White,
                        modifier = Modifier.size(30.dp)
                    )
                    Text(
                        if (on) "자동 ON" else "자동 OFF",
                        color = if (on) MaterialTheme.colorScheme.primary else Color.White, fontSize = 11.sp
                    )
                }
            }
        }
    }
}

private fun statusText(st: CaptureReadiness.Status?, auto: Boolean): String {
    if (st == null) return "책을 비춰 주세요"
    return when (st.issue) {
        CaptureReadiness.Issue.NO_BOOK -> "책을 비춰 주세요"
        CaptureReadiness.Issue.TOO_SMALL -> "책이 작아요 — 조금 더 가까이"
        CaptureReadiness.Issue.CUT_OFF -> "책이 잘렸어요 — 전체가 보이게 조금 멀리"
        CaptureReadiness.Issue.OFF_CENTER -> "책을 화면 가운데로"
        CaptureReadiness.Issue.TILTED -> "폰을 책과 나란히(수평으로) 들어 주세요"
        CaptureReadiness.Issue.DARK -> "어두워요 — 조명을 켜 보세요"
        CaptureReadiness.Issue.BRIGHT -> "너무 밝아요 — 빛 반사를 피해 주세요"
        CaptureReadiness.Issue.MOVING -> "잠시 멈춰 주세요"
        CaptureReadiness.Issue.BLURRY -> if (st.focusStuck) "화면을 탭해서 초점을 맞춰 주세요" else "초점을 맞추는 중…"
        CaptureReadiness.Issue.NONE -> if (auto) "좋아요 — 자동으로 찍어요" else "스캔하기 좋아요 — 찍으세요"
    }
}

@Composable
private fun OptionChip(label: String, selected: Boolean, onClick: () -> Unit) {
    FilterChip(
        selected = selected,
        onClick = onClick,
        label = { Text(label, fontSize = 13.sp) },
        modifier = Modifier.padding(end = 6.dp),
        colors = FilterChipDefaults.filterChipColors(
            labelColor = Color.White,
            selectedContainerColor = MaterialTheme.colorScheme.primary,
            selectedLabelColor = Color.Black,
        ),
    )
}
