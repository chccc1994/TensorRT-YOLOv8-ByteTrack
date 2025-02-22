cmake_minimum_required(VERSION 3.10)

project(yolov8)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(OpenCV_DIR "D:/Program/opencv3.4.16/build/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# ============= cuda ============
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(CMAKE_CUDA_ARCHITECTURES 86)  # 例如，对于 Ampere 架构的 GPU
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)
enable_language(CUDA)

# ============= tensorrt ============
# if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
#   message("Embed_platform on")
#   include_directories(/usr/include/aarch64-linux-gnu)
#   link_directories(/usr/lib/aarch64-linux-gnu)
# else()
message("WIN32")
set(TENSORRT_DIR "D:/Programs/TensorRT-8.6.1.6")
set(TENSORRT_INCLUDE_DIRS "${TENSORRT_DIR}/include")
set(TENSORRT_LIB_DIRS "${TENSORRT_DIR}/lib")
include_directories(${TENSORRT_INCLUDE_DIRS})
link_directories(${TENSORRT_DIR}/lib)
  # message("Embed_platform off")
  # include_directories(/usr/include/x86_64-linux-gnu)
  # link_directories(/usr/lib/x86_64-linux-gnu)
# endif()

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# set(OpenCV_DIR "D:/Program/opencv3.4.16/build")
# find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})


file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(yolov8 ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})

target_link_libraries(yolov8 nvinfer)
target_link_libraries(yolov8 cudart)
target_link_libraries(yolov8 myplugins)
target_link_libraries(yolov8 ${OpenCV_LIBS})

