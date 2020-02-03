cmake_minimum_required(VERSION 3.5)
project(NutLearning)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -march=native -Ofast -DNDEBUG")

include_directories(
    include
    /usr/local/include/
)

file(GLOB
    SRC_FILES
    src/*.cpp
)

add_executable(app ${SRC_FILES})