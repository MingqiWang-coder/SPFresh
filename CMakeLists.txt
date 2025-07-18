# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required (VERSION 3.12)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

project (SPTAGLib)

function(CXX_COMPILER_DUMPVERSION _OUTPUT_VERSION)
  exec_program(${CMAKE_CXX_COMPILER}
    ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
    OUTPUT_VARIABLE COMPILER_VERSION
  )

  set(${_OUTPUT_VERSION} ${COMPILER_VERSION} PARENT_SCOPE)
endfunction()

if(NOT WIN32)
    CXX_COMPILER_DUMPVERSION(CXX_COMPILER_VERSION)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # require at least gcc 5.0
    if (CXX_COMPILER_VERSION VERSION_LESS 5.0)
        message(FATAL_ERROR "GCC version must be at least 5.0!")
    endif()
    set (CMAKE_CXX_FLAGS "-Wall -Wunreachable-code -Wno-reorder -Wno-pessimizing-move -Wno-unused-result -Wno-strict-aliasing -Wno-sign-compare -Wno-unknown-pragmas -Wcast-align -lm -lrt -fopenmp -std=c++17")
    set (CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
    set (CMAKE_CXX_FLAGS_DEBUG   "-g -DDEBUG")

    find_path(NUMA_INCLUDE_DIR NAME numa.h
      HINTS $ENV{HOME}/local/include /opt/local/include /usr/local/include /usr/include)

    find_library(NUMA_LIBRARY NAME libnuma.so
      HINTS $ENV{HOME}/local/lib64 $ENV{HOME}/local/lib /usr/local/lib64 /usr/local/lib /opt/local/lib64 /opt/local/lib /usr/lib64 /usr/lib)

    find_library(NUMA_LIBRARY_STATIC NAME libnuma.a
      HINTS $ENV{HOME}/local/lib64 $ENV{HOME}/local/lib /usr/local/lib64 /usr/local/lib /opt/local/lib64 /opt/local/lib /usr/lib64 /usr/lib)

    if (NUMA_INCLUDE_DIR AND NUMA_LIBRARY AND NUMA_LIBRARY_STATIC)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lnuma")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lnuma")

        include_directories (${NUMA_INCLUDE_DIR})
        message (STATUS "Found numa library: inc=${NUMA_INCLUDE_DIR}, lib=${NUMA_LIBRARY}, staticlib=${NUMA_LIBRARY_STATIC}")
        set(Definition -DNUMA)
    else ()
        message (STATUS "WARNING: Numa library not found.")
        message (STATUS "Try: 'sudo yum install numactl numactl-devel' (or sudo apt-get install libnuma libnuma-dev)")
    endif ()

elseif(WIN32)
    if(NOT MSVC14)
         message(FATAL_ERROR "On Windows, only MSVC version 14 are supported!") 
    endif()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /c++17")

    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
else ()
    message(FATAL_ERROR "Unrecognized compiler (use GCC or MSVC)!")
endif()

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release CACHE STRING "Build types: Release Debug" FORCE)
endif()
message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (${CMAKE_SIZEOF_VOID_P} EQUAL "8")
    set (PROJECTNAME_ARCHITECTURE "x64")
else ()
    set (PROJECTNAME_ARCHITECTURE "x86")
endif ()
message (STATUS "Platform type: ${PROJECTNAME_ARCHITECTURE}")

set(Boost_USE_MULTITHREADED ON)

if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)

    set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    set (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR})
    set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR})
else()
    set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}/")
    set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}/")
endif()

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP)
if (OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message (STATUS "Found openmp.")
else()
    message (FATAL_ERROR "Could no find openmp!")
endif()

find_package(Boost 1.66 COMPONENTS system thread serialization wserialization regex filesystem)
if (Boost_FOUND)
    include_directories (${Boost_INCLUDE_DIR})
    link_directories (${Boost_LIBRARY_DIR})
    message (STATUS "Found Boost.")
    message (STATUS "Include Path: ${Boost_INCLUDE_DIRS}")
    message (STATUS "Library Path: ${Boost_LIBRARY_DIRS}")
    message (STATUS "Library: ${Boost_LIBRARIES}")
else()
    message (FATAL_ERROR "Could not find Boost >= 1.67!")
endif()

option(GPU "GPU" ON)
option(LIBRARYONLY "LIBRARYONLY" OFF)
option(ROCKSDB "ROCKSDB" ON)

if (ROCKSDB)
    add_definitions(-DROCKSDB)
endif()

add_subdirectory (ThirdParty/zstd/build/cmake)

add_subdirectory (AnnService)
add_subdirectory (Test)
#add_subdirectory (GPUSupport)
add_subdirectory (Wrappers)
add_subdirectory(bindings)
