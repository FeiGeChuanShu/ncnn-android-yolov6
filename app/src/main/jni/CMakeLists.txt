project(ncnnyolov6)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.3-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20220216-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(ncnnyolov6 SHARED yoloncnn.cpp yolo.cpp ndkcamera.cpp)

target_link_libraries(ncnnyolov6 ncnn ${OpenCV_LIBS} camera2ndk mediandk)
