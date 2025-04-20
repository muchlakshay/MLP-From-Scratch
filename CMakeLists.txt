cmake_minimum_required(VERSION 3.10)
project(MLP_From_Scratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/eigen)

add_executable(MLP
    main.cpp
    NeuralNetwork.cpp
)
