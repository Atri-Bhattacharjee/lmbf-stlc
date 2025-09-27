# --- CMake Configuration ---
cmake_minimum_required(VERSION 3.15)

# --- Vcpkg Integration ---
# Set the vcpkg toolchain file before the project() command to ensure CMake can find all libraries installed by vcpkg
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")

# --- Project Definition ---
project(smc_lmb_tracker LANGUAGES CXX)

# --- C++ Standard Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Project Dependencies ---
# Find required libraries managed by vcpkg
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# --- Python Module Target ---
# Create the Python module target named 'lmb_engine' that will be importable in Python
pybind11_add_module(lmb_engine src/main.cpp src/linear_propagator.cpp src/simple_sensor_model.cpp src/adaptive_birth_model.cpp)

# --- Include Directories ---
# Make the src directory available as an include directory
target_include_directories(lmb_engine PUBLIC src)

# --- Link Libraries ---
# Link the Eigen3 library to the Python module target
target_link_libraries(lmb_engine PRIVATE Eigen3::Eigen)