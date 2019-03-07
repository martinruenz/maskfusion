cmake_minimum_required(VERSION 3.1.0)

# c++ version
set(CMAKE_CXX_STANDARD 14)

# GCC warning / errors
if (CMAKE_COMPILER_IS_GNUCC)
  add_compile_options(-Wall -Wno-unused-function -Wno-unused-variable
    -Wno-unused-but-set-variable -Wno-unused-but-set-variable -Wno-write-strings
    -Wno-deprecated -Wno-deprecated-declarations -Werror -Wno-unknown-pragmas)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.1)
    add_compile_options(-Wno-misleading-indentation)
  endif()
endif()

# Don't follow symlinks when FILE GLOB_RECURSE (and don't warn)
cmake_policy(SET CMP0009 NEW)

project(MaskFusion)

set(BUILD_TESTS OFF CACHE BOOL "Build unit tests")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# boost
find_package(Boost 1.58.0 REQUIRED)
message("Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# gSLICr
set(GSLICR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/gSLICr")
include_directories(${GSLICR_DIR}/gSLICr_Lib)
link_directories(${GSLICR_DIR}/build)

# various algorithms
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/algorithms-cpp)

# Pangolin
find_package(Pangolin 0.4 REQUIRED)
message("Pangolin_INCLUDE_DIRS: ${Pangolin_INCLUDE_DIRS}")
include_directories(${Pangolin_INCLUDE_DIRS})

# Eigen3
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)
include_directories(${Eigen3_INCLUDE_DIRS})

# project-wide paths
include_directories(${PROJECT_SOURCE_DIR})

# -- CORE
add_subdirectory(Core)

# -- GUI
add_subdirectory(GUI)

# -- show all files
file(GLOB_RECURSE showfiles Core/* GUI/* Scripts/* README.md)
add_custom_target(showthings SOURCES ${showfiles})


export(
    TARGETS ${CMAKE_PROJECT_NAME}
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Targets.cmake"
)

configure_file(
    "CMakeModules/${CMAKE_PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake"
    COPYONLY
)
