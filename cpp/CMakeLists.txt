cmake_minimum_required(VERSION 3.0)
project(NutLearning)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} \
 -O3 -Wall -g -DNDEBUG -march=native -Ofast -std=c++14")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -g")

include_directories(
    /usr/local/include/
)
find_package(Eigen3 3.3 REQUIRED CONFIG)
add_executable(app main.cpp weight.cpp)
