# ONNX Runtime, OpenCV는 JNI로 호출되므로 난독화 대상에서 제외
-keep class ai.onnxruntime.** { *; }
-keep class org.opencv.** { *; }
