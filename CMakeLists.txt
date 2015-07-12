cmake_minimum_required(VERSION 3.2)
project(GKMeans)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMake")

## Compiler settings

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

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
    list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-std=c++11;-O2;-Xcompiler \"-fPIC\"")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()


#----------------------
#   Directory settings
#----------------------

include_directories(include)
cuda_include_directories(include)
set(SRC_DIR src/gkmeans)
set(TEST_SRC_DIR src/tests)

#----------------------
#   Library
#----------------------
FILE(GLOB SRC_FILES ${SRC_DIR}/*.cpp
                    ${SRC_DIR}/utils/*.cpp
                    ${SRC_DIR}/functions/*.cpp)

FILE(GLOB CU_SRC_FILES ${SRC_DIR}/*.cu
                       ${SRC_DIR}/utils/*.cu)

#CUDA_COMPILE(CU_OBJECT ${CU_SRC_FILES})

cuda_add_library(gkl SHARED ${SRC_FILES} ${CU_SRC_FILES})
target_link_libraries(gkl ${LIBRARIES})

#----------------------
#   EXE
#----------------------
add_executable(gkmeans tools/gkmeans.cpp)
target_link_libraries(gkmeans ${LIBRARIES} gkl)

#----------------------
#   TEST
#----------------------
FIND_PACKAGE(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

FILE(GLOB TEST_SRC_FILES ${TEST_SRC_DIR}/*.cpp)

add_executable(testAll ${TEST_SRC_FILES})
target_link_libraries(testAll ${LIBRARIES} gkl ${GTEST_LIBRARIES})
enable_testing()
add_test(NAME runtest COMMAND testAll)

#----------------------
#   DOC
#----------------------
#Find doxygen for generating API docs
find_package(Doxygen)
option(BUILD_DOC "Build documentation with doxygen")

if (BUILD_DOC)
    if (DOXYGEN_FOUND)
        message(STATUS "Found Doxygen version " ${DOXYGEN_VERSION})
        set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
        set(doxyfile ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)

        configure_file(${doxyfile_in} ${doxyfile} @ONLY)

        add_custom_target(doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generate API docs"
            VERBATIM)

        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/doc)

    else()
        message(FATAL_ERROR "Doxygen not found, documentation will not be built.")
    endif()
endif()