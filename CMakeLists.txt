cmake_minimum_required(VERSION 3.21)
project(project_empire_2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(project_empire_2 main.cu)

set_target_properties(project_empire_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
