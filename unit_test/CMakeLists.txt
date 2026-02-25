cmake_minimum_required(VERSION 3.12.0)

project(rabitq_unit_tests LANGUAGES CXX)

include(FetchContent)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# Try to use local googletest first, then fallback to FetchContent
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest")
    message(STATUS "Using local googletest from third_party directory")
    add_subdirectory(third_party/googletest)
else()
    message(STATUS "Local googletest not found, downloading via FetchContent...")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0 
    )
    FetchContent_MakeAvailable(googletest)
endif()

EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -fprefetch-loop-arrays -march=armv8.2-a+lse+fp16+dotprod -mtune=tsv110")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage")

find_package(HDF5 REQUIRED COMPONENTS C HL)
include_directories(${HDF5_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/../src)
# Check if Eigen directory exists
if(EXISTS "${PROJECT_SOURCE_DIR}/../src/Eigen")
    message(STATUS "Found Eigen in src/Eigen")
else()
    message(WARNING "Eigen not found in src/Eigen. Please download Eigen 3.4.0 and place it in src/Eigen")
endif()

#add_executable(rabitq_unit_test
#    main_test.cpp
#    construction_test.cpp
#    loadsave_test.cpp
#    search_test.cpp
#)

add_executable(rabitq_unit_test
    main_test.cpp
    rabitq_test.cpp
    ../src/test_result.cpp
)

target_compile_options(rabitq_unit_test PRIVATE
    -O3
    -g
    -D FAST_SCAN
    -DnumC=128
    -fprefetch-loop-arrays
    -march=armv8.2-a+lse+fp16+dotprod+fp16fml
    -mtune=tsv110
    -fprofile-arcs
    -ftest-coverage
    -fsanitize=address
)

# Note: numC is defined per test case, not globally
target_compile_definitions(rabitq_unit_test PRIVATE B_QUERY=4)

target_link_libraries(rabitq_unit_test 
    PRIVATE 
    gtest 
    gtest_main 
    ${HDF5_LIBRARIES} 
    --coverage
    -fsanitize=address
)

enable_testing()
add_test(NAME rabitq_unit_tests COMMAND rabitq_unit_test)
