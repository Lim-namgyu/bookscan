package com.bookscan.app

import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import com.bookscan.app.ui.CameraScreen
import com.bookscan.app.ui.PageScreen
import com.bookscan.app.ui.ReviewScreen

sealed interface Screen {
    data object Camera : Screen
    data object Review : Screen
    data class Page(val id: Long) : Screen
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Color(0xFFF2B84B),
                    onPrimary = Color(0xFF1F1A10),
                    background = Color.Black,
                    surface = Color(0xFF121212),
                )
            ) {
                val vm: MainViewModel = viewModel()
                var screen by remember { mutableStateOf<Screen>(Screen.Camera) }
                val ctx = LocalContext.current

                LaunchedEffect(vm.message) {
                    vm.message?.let {
                        Toast.makeText(ctx, it, Toast.LENGTH_LONG).show()
                        vm.message = null
                    }
                }

                BackHandler(enabled = screen != Screen.Camera) {
                    screen = when (screen) {
                        is Screen.Page -> Screen.Review
                        else -> Screen.Camera
                    }
                }

                when (val s = screen) {
                    Screen.Camera -> CameraScreen(vm, onOpenReview = { screen = Screen.Review })
                    Screen.Review -> ReviewScreen(
                        vm,
                        onBack = { screen = Screen.Camera },
                        onOpenPage = { screen = Screen.Page(it) },
                    )
                    is Screen.Page -> {
                        val page = vm.pages.firstOrNull { it.id == s.id }
                        if (page == null) {
                            LaunchedEffect(Unit) { screen = Screen.Review }
                        } else {
                            PageScreen(vm, page, onBack = { screen = Screen.Review })
                        }
                    }
                }
            }
        }
    }
}
