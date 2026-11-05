package com.bookscan.app.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import com.bookscan.app.data.ImageIO
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/** 파일 이미지를 줄여서 비동기로 표시. version 이 바뀌면 다시 읽는다. */
@Composable
fun PageImage(
    file: File,
    version: Int,
    maxSide: Int,
    modifier: Modifier = Modifier,
    contentScale: ContentScale = ContentScale.Fit,
) {
    val bmp by produceState<ImageBitmap?>(initialValue = null, file, version) {
        value = withContext(Dispatchers.IO) { ImageIO.decodeSampled(file, maxSide)?.asImageBitmap() }
    }
    val b = bmp
    if (b != null) {
        // FilterQuality.High: 큰 이미지를 화면 크기로 줄여 보일 때 계단·울퉁불퉁함 없이 매끈하게(밉맵/고품질 보간)
        Image(bitmap = b, contentDescription = null, modifier = modifier, contentScale = contentScale, filterQuality = FilterQuality.High)
    } else {
        Box(modifier.background(Color(0xFF2A2A2A)))
    }
}
