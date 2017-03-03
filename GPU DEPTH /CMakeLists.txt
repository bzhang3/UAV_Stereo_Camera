SET(execName ZED_background_subtraction)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

SET(VERSION_REQ_OCV "3.1")
SET(VERSION_REQ_CUDA "8.0")

IF(WIN32) # Windows
    SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
    SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
    SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
    SET(OPENCV_DIR $ENV{OPENCV_DIR})
    SET(EIGEN3_INCLUDE_DIR $ENV{EIGEN_DIR})
ELSE() # Linux
    find_package(ZED REQUIRED)
    find_package(Eigen3 REQUIRED)

##For Jetson, OpenCV4Tegra is based on OpenCV2.4
     exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
     if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" OR CMAKE_SYSTEM_NAME2 MATCHES "armv7l" )
	SET(VERSION_REQ_OCV "2.4")
	SET(VERSION_REQ_CUDA "")
     endif()
ENDIF(WIN32)

find_package(OpenCV ${VERSION_REQ_OCV} REQUIRED)
find_package(CUDA ${VERSION_REQ_CUDA} REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")
FILE(GLOB_RECURSE SRC_CU_FOLDER "${SRC_FOLDER}/*.cu")

list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_32)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_52)

##Check x86/K1/X1
exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)

if (NOT CMAKE_SYSTEM_NAME2 MATCHES "armv7l")
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_53) 
endif()

### Adding 60/61 CC if CUDA 8.0
if (CUDA_VERSION MATCHES "8.0")
message(" --> Adding CC : 6.0 / 6.1 ")
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_61)
endif()

### New command add

unset(CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
message("CUDA_USE_STATIC_CUDA_RUNTIME : ${CUDA_USE_STATIC_CUDA_RUNTIME}")



set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
cuda_add_executable(${execName} ${SRC_FILES} ${SRC_CU_FOLDER})

set_property(TARGET ${execName} PROPERTY OUTPUT_NAME "ZED background subtraction")
add_definitions(-std=c++11 -g -O3)

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
			${ZED_LIBRARIES}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY}
			)
