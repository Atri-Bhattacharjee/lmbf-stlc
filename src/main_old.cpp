


#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <memory>
#include "linear_propagator.h"
#include "datatypes.h"
#include "models.h"
#include "simple_sensor_model.h"
#include "adaptive_birth_model.h"
#include "smc_lmb_tracker.h"





PYBIND11_MODULE(lmb_engine, m) {
    pybind11::class_<Particle>(m, "Particle")
        .def(pybind11::init<>())
        .def_readwrite("state_vector", &Particle::state_vector)
        .def_readwrite("weight", &Particle::weight);

    pybind11::class_<Measurement>(m, "Measurement")
        .def(pybind11::init<>())
        .def_readwrite("timestamp_", &Measurement::timestamp_)
        .def_readwrite("value_", &Measurement::value_)
        .def_readwrite("covariance_", &Measurement::covariance_)
        .def_readwrite("sensor_id_", &Measurement::sensor_id_);

    pybind11::class_<TrackLabel>(m, "TrackLabel")
        .def(pybind11::init<>())
        .def_readwrite("birth_time", &TrackLabel::birth_time)
        .def_readwrite("index", &TrackLabel::index);

    pybind11::class_<Track>(m, "Track")
        .def(pybind11::init<>())
        .def(pybind11::init<const TrackLabel&, double, const std::vector<Particle>&>())
        .def("label", &Track::label, pybind11::return_value_policy::reference_internal)
        .def("existence_probability", &Track::existence_probability)
        .def("particles", &Track::particles, pybind11::return_value_policy::reference_internal)
        .def("set_existence_probability", &Track::set_existence_probability)
        .def("set_particles", &Track::set_particles);

    pybind11::class_<FilterState>(m, "FilterState")
        .def(pybind11::init<>())
        .def(pybind11::init<double, const std::vector<Track>&>())
        .def("timestamp", &FilterState::timestamp)
        .def("tracks", &FilterState::tracks, pybind11::return_value_policy::reference_internal)
        .def("set_timestamp", &FilterState::set_timestamp)
        .def("set_tracks", &FilterState::set_tracks);

    pybind11::class_<IOrbitPropagator, std::shared_ptr<IOrbitPropagator>>(m, "IOrbitPropagator");

    pybind11::class_<LinearPropagator, IOrbitPropagator, std::shared_ptr<LinearPropagator>>(m, "LinearPropagator")
        .def(pybind11::init<>())
        .def("propagate", &LinearPropagator::propagate);

    // ...existing sensor model bindings remain for now...
    pybind11::class_<ISensorModel, std::shared_ptr<ISensorModel>>(m, "ISensorModel");

    pybind11::class_<SimpleSensorModel, ISensorModel, std::shared_ptr<SimpleSensorModel>>(m, "SimpleSensorModel")
        .def(pybind11::init<>())
        .def("calculate_likelihood", &SimpleSensorModel::calculate_likelihood);

    pybind11::class_<IBirthModel, std::shared_ptr<IBirthModel>>(m, "IBirthModel");

    pybind11::class_<AdaptiveBirthModel, IBirthModel, std::shared_ptr<AdaptiveBirthModel>>(m, "AdaptiveBirthModel")
        .def(pybind11::init<int, double, const Eigen::MatrixXd&>())
        .def("generate_new_tracks", &AdaptiveBirthModel::generate_new_tracks);

    pybind11::class_<SMC_LMB_Tracker, std::shared_ptr<SMC_LMB_Tracker>>(m, "SMC_LMB_Tracker")
        .def(pybind11::init<>())
        .def(pybind11::init<std::shared_ptr<IOrbitPropagator>, std::shared_ptr<ISensorModel>, std::shared_ptr<IBirthModel>, double>())
        .def("predict", &SMC_LMB_Tracker::predict)
        .def("update", &SMC_LMB_Tracker::update)
        .def("get_tracks", &SMC_LMB_Tracker::get_tracks, pybind11::return_value_policy::reference_internal)
        .def("set_tracks", &SMC_LMB_Tracker::set_tracks);
}