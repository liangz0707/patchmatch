cmake_minimum_required(VERSION 3.6)
project(patch_match)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp src/patchmatch.cpp src/patchmatch.h)

# 添加opencv依赖
include_directories(/usr/local/Cellar/opencv/2.4.13.1/include)
link_directories(/usr/local/Cellar/opencv/2.4.13.1/lib)
link_libraries(libopencv_imgproc.dylib libopencv_highgui.dylib libopencv_core.dylib libopencv_features2d.dylib)

add_executable(patch_match ${SOURCE_FILES})