cmake_minimum_required(VERSION 3.10)

project(yolov8_bytetrack)

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE release)

# eigen3
set(Eigen3_INCLUDE_DIRS "D:/Programs/Eigen3/include/eigen3")
include_directories(${Eigen3_INCLUDE_DIRS})

set(OpenCV_DIR "D:/Program/opencv3.4.16/build/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-std=c++17)
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
set(TENSORRT_DIR "D:/Programs/TensorRT-8.6.1.6")
set(TENSORRT_INCLUDE_DIRS "${TENSORRT_DIR}/include")
set(TENSORRT_LIB_DIRS "${TENSORRT_DIR}/lib")
include_directories(${TENSORRT_INCLUDE_DIRS})
link_directories(${TENSORRT_DIR}/lib)

# =========== bytetrack lib ===========
include_directories(${PROJECT_SOURCE_DIR}/bytetrack/include)
file(GLOB_RECURSE SRCS01 ${PROJECT_SOURCE_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${SRCS01})
# target_link_libraries(bytetrack cudart nvinfer ${OpenCV_LIBS})
target_link_libraries(bytetrack cudart nvinfer ${OpenCV_LIBS})
set_target_properties(bytetrack PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(bytetrack PROPERTIES
    LINKER_LANGUAGE CXX
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
# ============= yolov8 lib =============
include_directories(${PROJECT_SOURCE_DIR}/yolo/include)
include_directories(${PROJECT_SOURCE_DIR}/yolo/plugin)
file(GLOB_RECURSE SRCS02 
                ${PROJECT_SOURCE_DIR}/yolo/plugin/*.cu                
                ${PROJECT_SOURCE_DIR}/yolo/src/*.cpp 
                ${PROJECT_SOURCE_DIR}/yolo/src/*.cu 
)
message(${SRCS02})
cuda_add_library(yolo_infer SHARED ${SRCS02})
target_link_libraries(yolo_infer 
                    ${OpenCV_LIBS}
                    nvinfer
                    nvinfer_plugin
                    cudart
                    nvonnxparser
)

# ======== main execute file ========
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main bytetrack yolo_infer)
