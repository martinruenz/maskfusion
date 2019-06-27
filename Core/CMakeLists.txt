cmake_minimum_required(VERSION 3.0.2)

# policy to export a different link interface
cmake_policy(SET CMP0022 OLD)

project(libmaskfusion)

message("Evaluating Core/CMAKE")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

## External packages
find_package(CUDA REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(OpenCV 3.1 REQUIRED)
find_package(OpenMP)
find_package(PythonLibs)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

## Options
set(MASKFUSION_GPUS_MASKRCNN "0" CACHE STRING "Select GPU which is used by MaskRCNN")
set(MASKFUSION_GPU_SLAM "-1" CACHE STRING "Select GPU which is running SLAM, this has to be the OpenGL GPU.")
set(MASKFUSION_NUM_GSURFELS "9437184" CACHE STRING "Maximum number of surfels for background")
set(MASKFUSION_NUM_OSURFELS "1048576" CACHE STRING "Maximum number of surfels per object")
add_definitions(-DMASKFUSION_GPUS_MASKRCNN=${MASKFUSION_GPUS_MASKRCNN})
add_definitions(-DMASKFUSION_GPU_SLAM=${MASKFUSION_GPU_SLAM})
add_definitions(-DMASKFUSION_NUM_GSURFELS=${MASKFUSION_NUM_GSURFELS})
add_definitions(-DMASKFUSION_NUM_OSURFELS=${MASKFUSION_NUM_OSURFELS})

# Show debug visualisations
option(SHOW_DEBUG_VISUALISATION "Show debug visualisation" OFF)
if(SHOW_DEBUG_VISUALISATION)
    add_definitions(-DSHOW_DEBUG_VISUALISATION)
endif()

## Directories
set(DENSECRF_DIR "${CMAKE_SOURCE_DIR}/deps/densecrf")
set(DENSECRF_LIBRARY_DIRS "${DENSECRF_DIR}/build/src/")
set(DENSECRF_INCLUDE_DIRS "${DENSECRF_DIR}/include")
set(DENSECRF_LIBRARIES densecrf)
set(maskfusion_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")

message("Using: ")
message("=======")
message("Using DENSECRF_DIR: ${DENSECRF_DIR}")
message("Using maskfusion_SHADER_DIR: ${maskfusion_SHADER_DIR}")
message("Using python:  ${PYTHONLIBS_VERSION_STRING} ")
message("Using PYTHON_LIBRARIES:  ${PYTHON_LIBRARIES} ")
message("Using PYTHON_INCLUDE_DIRS:  ${PYTHON_INCLUDE_DIRS} ")
message("=======")

## Includes
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories(${DENSECRF_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
link_directories(${DENSECRF_LIBRARY_DIRS})

file(GLOB srcs *.cpp)
file(GLOB model_srcs Model/*.cpp)
file(GLOB utils_srcs Utils/*.cpp)
file(GLOB shader_srcs Shaders/*.cpp)
file(GLOB cuda Cuda/*.cu)
file(GLOB containers Cuda/containers/*.cpp)
file(GLOB segmentation Segmentation/*.cpp Segmentation/MaskRCNN/*.cpp)

set(CUDA_ARCH_BIN "50 52 61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-D_FORCE_INLINES;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-g")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

CUDA_COMPILE(cuda_objs ${cuda})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse -msse2 -msse3 -DSHADER_DIR=${maskfusion_SHADER_DIR}")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug build.")
else(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release build.")
    add_compile_options(-O3)
    add_compile_options(-Wno-maybe-uninitialized)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

add_library(${PROJECT_NAME} SHARED
            ${srcs}
            ${model_srcs}
            ${utils_srcs}
            ${shader_srcs}
            ${cuda}
            ${cuda_objs}
            ${containers}
            ${segmentation}
)

target_include_directories(${PROJECT_NAME} PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}
                            ${GSLICR_DIR}/gSLICr_Lib
                            ${ZLIB_INCLUDE_DIR}
                            ${Pangolin_INCLUDE_DIRS}
                            ${CUDA_INCLUDE_DIRS}
                            ${OPENNI2_INCLUDE_DIR}
                            ${OpenCV_INCLUDE_DIRS}
                            ${BOOST_INCLUDE_DIRS}
                            ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
                      ${DENSECRF_LIBRARIES}
                      ${Eigen_LIBRARIES}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${SUITESPARSE_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      gSLICr_lib
                      ${PYTHON_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES LINK_INTERFACE_LIBRARIES "")



target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

export(PACKAGE ${PROJECT_NAME})


export(TARGETS ${PROJECT_NAME}
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
)

configure_file(
    "../CMakeModules/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    COPYONLY
)
