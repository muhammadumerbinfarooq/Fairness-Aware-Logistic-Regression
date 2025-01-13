cmake_minimum_required(VERSION 3.10)
project(FairnessAwareLogisticRegression)

set(CMAKE_CXX_STANDARD 14)

# Include directories
include_directories(include)

# Source files
set(SOURCES src/fair_logistic_regression.cpp)

# Add executable
add_executable(fairness_logistic_regression ${SOURCES})
