// 엔진(engine/*.kt)을 PC(JVM) 에서 그대로 돌려 보는 검사용 프로젝트.
// 앱과 같은 소스를 컴파일하고, 데스크톱용 OpenCV(openpnp 패키징)·ONNX Runtime 으로 실행한다.
// GitHub Actions(engine-test.yml) 에서 파이썬 기준 구현과 결과를 비교한다.
plugins {
    kotlin("jvm") version "2.1.0"
    application
}

repositories { mavenCentral() }

dependencies {
    implementation("org.openpnp:opencv:4.9.0-0")
    implementation("com.microsoft.onnxruntime:onnxruntime:1.20.0")
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

sourceSets {
    main {
        kotlin.srcDir("../BookScan/app/src/main/java/com/bookscan/app/engine")
    }
}

application {
    mainClass.set("enginetest.MainKt")
    applicationDefaultJvmArgs = listOf("-Xmx2g")
}
