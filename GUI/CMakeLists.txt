cmake_minimum_required(VERSION 2.6.0)

project(maskfusion-gui)

message("Evaluating GUI/CMAKE")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(ZLIB REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(OpenCV 3.1 REQUIRED )
find_package(Boost 1.58.0 REQUIRED)

option(WITH_FREENECT2 "Enable freenect support" OFF)
if(WITH_FREENECT2)
    find_package(freenect2 REQUIRED)
    add_definitions(-DWITH_FREENECT2)
    list(APPEND LIBRARIES ${freenect2_LIBRARIES})
endif()

option(WITH_FREETYPE_GL_CPP "Enable drawing of labels" OFF)
if(WITH_FREETYPE_GL_CPP)
#    find_package(freetype-gl-cpp REQUIRED ${CMAKE_SOURCE_DIR}/deps/freetype-gl-cpp/build)
    link_directories(${CMAKE_SOURCE_DIR}/deps/freetype-gl-cpp/install/lib/freetype-gl-cpp)
    include_directories(${CMAKE_SOURCE_DIR}/deps/freetype-gl-cpp/install/include)
    add_definitions(-DWITH_FREETYPE_GL_CPP)
    list(APPEND LIBRARIES "freetype-gl-cpp")
    message("WITH_FREETYPE_GL_CPP")
endif()

include_directories(${CMAKE_SOURCE_DIR}/Core)
include_directories(${CMAKE_SOURCE_DIR}/deps/toml11)
include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

message("Include-directories: ")
message("ZLIB_INCLUDE_DIR: ${ZLIB_INCLUDE_DIR}")
message("EIGEN_INCLUDE_DIRS: ${EIGEN_INCLUDE_DIRS}")
message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message("OPENNI2_INCLUDE_DIR: ${OPENNI2_INCLUDE_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR/../Core: ${${CMAKE_CURRENT_SOURCE_DIR}/../Core}")

set(MASKFUSION_MASK_RCNN_DIR ${CMAKE_SOURCE_DIR}/deps/Mask_RCNN CACHE STRING "Location of MaskRCNN")
set(MASKFUSION_PYTHON_VE_PATH "" CACHE STRING "Python virtual environment to use")
set(MASKFUSION_MODEL_FILE "${MASKFUSION_MASK_RCNN_DIR}/data/mask_rcnn_coco.h5" CACHE STRING "H5 MaskRCNN model weights")
configure_file("../Core/Segmentation/MaskRCNN/MaskRCNN.py.in" "${CMAKE_CURRENT_BINARY_DIR}/MaskRCNN.py")
configure_file("../Core/Segmentation/MaskRCNN/helpers.py" "${CMAKE_CURRENT_BINARY_DIR}/helpers.py")
configure_file("../config.toml.in" "${CMAKE_CURRENT_BINARY_DIR}/config.toml")

file(GLOB srcs *.cpp)
file(GLOB tools_srcs Tools/*.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse -msse2 -msse3")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug build.")
else(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release build.")
    add_compile_options(-O3)
    add_compile_options(-Wno-maybe-uninitialized)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

add_definitions(-Dlinux=1)

add_executable(MaskFusion
               ${srcs}
               ${tools_srcs}
)

target_link_libraries(MaskFusion
                      libmaskfusion
                      ${ZLIB_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${OPENNI2_LIBRARY}
                      ${OpenCV_LIBRARIES}
                      ${LIBRARIES}
                      boost_filesystem
                      boost_system
                      pthread
                      jpeg
)

add_library(MaskFusionTools SHARED
            ${tools_srcs})

target_link_libraries(MaskFusionTools
                      ${ZLIB_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${OPENNI2_LIBRARY}
                      ${OpenCV_LIBRARIES}
                      ${freenect2_LIBRARIES}
                      boost_filesystem
                      boost_system
                      pthread
)

target_include_directories(MaskFusionTools PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}
                            ${ZLIB_INCLUDE_DIR}
                            ${Pangolin_INCLUDE_DIRS}
                            ${CUDA_INCLUDE_DIRS}
                            ${OPENNI2_INCLUDE_DIR}
                            ${OpenCV_INCLUDE_DIRS}
                            ${BOOST_INCLUDE_DIRS}
                            ${freenect2_INCLUDE_DIR}
)
