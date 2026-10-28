plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.bookscan.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.bookscan.app"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // 휴대폰 CPU(64비트 ARM)만 포함해 APK 용량을 줄임. 에뮬레이터 테스트 시 "x86_64" 추가
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    // 개인 설치용 고정 서명 키. 같은 키로 서명해야 새 버전을 지우지 않고 덮어 설치할 수 있다.
    // (Play 스토어 배포 시에는 별도 키를 만들어 저장소 밖에 보관할 것)
    signingConfigs {
        create("personal") {
            storeFile = file("bookscan.keystore")
            storePassword = "bookscan123"
            keyAlias = "bookscan"
            keyPassword = "bookscan123"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("personal")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }

    // 모델 파일은 압축하지 않고 그대로 넣는다(로딩 빠름)
    androidResources { noCompress += listOf("onnx") }

    // 개인 설치용 빌드에서 경고성 검사 때문에 빌드가 멈추지 않게
    lint {
        checkReleaseBuilds = false
        abortOnError = false
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // 카메라
    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // 영상 처리 / 곡면 펴기 모델
    implementation("org.opencv:opencv:4.10.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
