#include <pybind11/pybind11.h>

class TestClass {
public:
    TestClass(int x) : value(x) {}
    int get_value() const { return value; }
private:
    int value;
};

PYBIND11_MODULE(pybind11_test, m) {
    pybind11::class_<TestClass>(m, "TestClass")
        .def(pybind11::init<int>())
        .def("get_value", &TestClass::get_value);
}
