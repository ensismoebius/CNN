cmake_minimum_required(VERSION 3.9)
project(TesteNeuro
        VERSION 0.1.0
        DESCRIPTION "Multilayer Neural Network toy"
        LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (APPLE)
	set (CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING
    "Minimum OS X deployment version" FORCE)
	
	# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
	if (POLICY CMP0025)
		cmake_policy(SET CMP0025 NEW)
	endif ()

    # If Apple
    set(BLA_VENDOR Apple)
else()
    # If another OSs
    set(BLA_VENDOR OpenBLAS)
endif()

# For matrices operations
find_package(Armadillo REQUIRED)

# For paralelism
find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP)

set (SRCS
src/main.cpp
src/lib/NN.cpp
src/lib/inpuLayer.cpp
)

# Include dirs
include_directories(${ARMADILLO_INCLUDE_DIRS})

if(OpenMP_CXX_FOUND)
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

# Executable
add_executable(${CMAKE_PROJECT_NAME} ${SRCS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})
