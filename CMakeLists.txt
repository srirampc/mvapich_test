cmake_minimum_required(VERSION 3.1.0)

set(PROJECT_NAME_STR test_mvapich2)
PROJECT(${PROJECT_NAME_STR} C CXX)
#TODO CHANGE THIS OPTION
set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# flags
set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -std=c++11 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -std=c++11")

include_directories("${PROJECT_SOURCE_DIR}/ext/mxx/include")

#
# add MPI includes and libraries
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
message(STATUS "Found MPI:")
message(STATUS "    headers: ${MPI_INCLUDE_PATH}")
message(STATUS "    libs:    ${MPI_LIBRARIES}")
message(STATUS "    link flags: ${MPI_LINK_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} ${MPI_LINK_FLAGS}")

# Compile
set(TESTMVA2A_EXEC ${PROJECT_NAME_STR}_all2allv.x)
ADD_EXECUTABLE(${TESTMVA2A_EXEC} src/test_mva2av.cpp)
target_link_libraries(${TESTMVA2A_EXEC} "${MPI_LIBRARIES}")
