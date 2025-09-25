#include <pybind11/pybind11.h>
#include "datatypes.h"

PYBIND11_MODULE(lmb_engine, m) {
    m.doc() = "High-performance C++ engine for SMC-LMB space debris tracking";
    
    // --- Bindings for core data structures will go here ---
}