package com.bookscan.app.data

import com.bookscan.app.engine.ColorMode
import com.bookscan.app.engine.Side
import java.io.File

/**
 * 스캔한 페이지 한 장.
 * flatFile  : 곡면만 편 이미지(보정 전) — 모드 변경·손가락 제거 on/off 시 여기서 다시 보정
 * resultFile: 최종 결과 이미지
 * version   : 파일이 바뀔 때마다 증가(화면 이미지 캐시 갱신용)
 */
data class ScanPage(
    val id: Long,
    val flatFile: File,
    val resultFile: File,
    val skipSide: Side?,
    val mode: ColorMode,
    val removeFingers: Boolean,
    val version: Int = 0,
)
