cmake_minimum_required(VERSION 3.1.0)
project(sequence_add_bbox_labels)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/../deps/opencv/build)
find_package(OpenCV 3.4.0 REQUIRED)
find_package(Boost 1.58.0 COMPONENTS filesystem system REQUIRED)

add_executable(${PROJECT_NAME} sequence_add_bbox_labels.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
include_directories(${Boost_INCLUDE_DIR})
