#cmake_minimum_required(VERSION 2.8.12)
cmake_minimum_required(VERSION 3.0)
project(dynamics)

find_package(Eigen3 REQUIRED)

include_directories(
        /home/dongwon/research/Euclidean_planning/include
        /home/dongwon/research/Euclidean_planning
        ${EIGEN3_INCLUDE_DIR}
)

add_subdirectory(pybind11)
pybind11_add_module(DynamicsCpp src/main.cpp)
