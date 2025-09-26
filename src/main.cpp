#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include "datatypes.h"

PYBIND11_MODULE(lmb_engine, m) {
    m.doc() = "High-performance C++ engine for SMC-LMB space debris tracking";
    

    // Example function that uses Eigen matrix
    m.def("create_identity_matrix", [](int size) {
        return Eigen::MatrixXd::Identity(size, size);
    });
    
    // Test function for datatypes - create a simple TrackLabel
    m.def("create_test_track_label", []() {
        TrackLabel label;
        label.birth_time = 123456789;
        label.index = 42;
        return std::make_pair(label.birth_time, label.index);
    });
}