// Basic pybind11 module structure for the lmb_engine Python module
// This file will be the main entry point for the Python module

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

// Define the Python module
PYBIND11_MODULE(lmb_engine, m) {
    m.doc() = "LMB Tracker Engine - Python bindings for the LMB tracking algorithm";
    
    // Example function that uses Eigen - this demonstrates the integration works
    m.def("hello_world", []() {
        return "Hello from LMB Engine!";
    });
    
    // Example function that uses Eigen matrix
    m.def("create_identity_matrix", [](int size) {
        return Eigen::MatrixXd::Identity(size, size);
    });
}