cmake_minimum_required(VERSION 3.2)
project(GKMeans)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMake")

## Compiler settings

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O2")

enable_testing()

#----------------------
#   Dependencies
#----------------------

FIND_PACKAGE(Glog REQUIRED)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND LIBRARIES ${GLOG_LIBRARIES})

FIND_PACKAGE(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA Version: " ${CUDA_VERSION_STRINGS})
    message(STATUS "CUDA Libararies: " ${CUDA_LIBRARIES})
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
endif()

#----------------------
#   Directory settings
#----------------------

include_directories(include)
set(SRC_DIR src/gkmeans)

#----------------------
#   Library
#----------------------
FILE(GLOB SRC_FILES ${SRC_DIR}/*.cpp)

add_library(gkl SHARED ${SRC_FILES})
target_link_libraries(gkl ${LIBRARIES})

#----------------------
#   EXE
#----------------------
add_executable(gkmeans tools/gkmeans.cpp)
target_link_libraries(gkmeans ${LIBRARIES} gkl)