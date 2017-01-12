cmake_minimum_required(VERSION 3.6)
project(patch_match)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp src/patchmatch.cpp src/patchmatch.h src/LocalPatchMatch.cpp src/LocalPatchMatch.h)

# 添加opencv依赖
include_directories(/usr/local/Cellar/opencv/2.4.13.1/include)
link_directories(/usr/local/Cellar/opencv/2.4.13.1/lib)
link_libraries(libopencv_imgproc.dylib libopencv_highgui.dylib libopencv_core.dylib libopencv_features2d.dylib)

#动态链接库文件.dylib
#add_library(patch_match SHARED ${SOURCE_FILES})
#静态链接库文件.a
#add_library(patch_match ${SOURCE_FILES})
#可执行文件.exe 需要有main方法
add_executable(patch_match ${SOURCE_FILES})
