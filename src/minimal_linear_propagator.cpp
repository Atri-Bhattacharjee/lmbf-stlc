#include <pybind11/pybind11.h>

class LinearPropagator {
public:
    LinearPropagator() = default;
    int test_method() const { return 123; }
};

PYBIND11_MODULE(minimal_linear_propagator, m) {
    pybind11::class_<LinearPropagator>(m, "LinearPropagator")
        .def(pybind11::init<>())
        .def("test_method", &LinearPropagator::test_method);
}
