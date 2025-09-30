# --- CMake Configuration ---
cmake_minimum_required(VERSION 3.15)

# --- Project Definition ---
project(smc_lmb_tracker LANGUAGES CXX)

# --- C++ Standard Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Project Dependencies ---
find_package(Eigen3 REQUIRED)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_SOURCE_DIR}/external/vcpkg/installed/x64-windows/share/pybind11")
find_package(pybind11 CONFIG REQUIRED)

# --- Python Module Target ---
# Create the Python module target named 'lmb_engine' that will be importable in Python
pybind11_add_module(lmb_engine 
    src/main.cpp 
    src/linear_propagator.cpp 
    src/simple_sensor_model.cpp
    src/adaptive_birth_model.cpp
    src/smc_lmb_tracker.cpp
)

# --- Minimal Test Python Module Target ---
pybind11_add_module(pybind11_test src/pybind11_test.cpp)
set_target_properties(pybind11_test PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

# --- Minimal Linear Propagator Test Module ---
pybind11_add_module(minimal_linear_propagator src/minimal_linear_propagator.cpp)
set_target_properties(minimal_linear_propagator PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/minimal_linear_propagator"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/minimal_linear_propagator"
)

# --- Include Directories ---
target_include_directories(lmb_engine PUBLIC src)

# --- Link Libraries ---
target_link_libraries(lmb_engine PRIVATE Eigen3::Eigen)

# --- Set Output Directory for Python Package ---
set_target_properties(lmb_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/lmb_engine/Debug"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/lmb_engine/Release"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/lmb_engine/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/lmb_engine/Release"
)