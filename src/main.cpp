#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "datatypes.h"
#include "models.h"
#include "linear_propagator.h"
#include "simple_sensor_model.h"
#include "adaptive_birth_model.h"
#include "smc_lmb_tracker.h"

PYBIND11_MODULE(lmb_engine, m) {
    m.doc() = "High-performance C++ engine for SMC-LMB space debris tracking";
    
    // Bind core data structures
    pybind11::class_<TrackLabel>(m, "TrackLabel")
        .def(pybind11::init<>())
        .def_readwrite("birth_time", &TrackLabel::birth_time)
        .def_readwrite("index", &TrackLabel::index);
    
    pybind11::class_<Particle>(m, "Particle")
        .def(pybind11::init<>())
        .def_readwrite("state_vector", &Particle::state_vector)
        .def_readwrite("weight", &Particle::weight);
    
    pybind11::class_<Track>(m, "Track")
        .def(pybind11::init<const TrackLabel&, double, const std::vector<Particle>&>())
        .def_property("label", &Track::label, nullptr)
        .def_property("existence_probability", &Track::existence_probability, &Track::set_existence_probability)
        .def_property("particles", &Track::particles, &Track::set_particles);
    
    // Bind concrete model implementations
    pybind11::class_<LinearPropagator>(m, "LinearPropagator")
        .def(pybind11::init<>());
    
    pybind11::class_<SimpleSensorModel>(m, "SimpleSensorModel")
        .def(pybind11::init<>());
    
    pybind11::class_<AdaptiveBirthModel>(m, "AdaptiveBirthModel")
        .def(pybind11::init<int, double, const Eigen::MatrixXd&>());
    
    // Bind the main tracker class with a custom factory function
    pybind11::class_<SMC_LMB_Tracker>(m, "SMC_LMB_Tracker")
        .def("predict", &SMC_LMB_Tracker::predict, "Runs the predict step for a given time delta")
        .def("get_tracks", &SMC_LMB_Tracker::get_tracks, "Gets the current list of tracks")
        .def("set_tracks", &SMC_LMB_Tracker::set_tracks, "Sets the initial list of tracks for the filter");
    
    // Factory function to create SMC_LMB_Tracker with proper unique_ptr management
    m.def("create_smc_lmb_tracker", [](double survival_probability) {
        auto propagator = std::make_unique<LinearPropagator>();
        auto sensor_model = std::make_unique<SimpleSensorModel>();
        Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(7, 7) * 0.1;  // Default covariance
        auto birth_model = std::make_unique<AdaptiveBirthModel>(100, 0.1, covariance);  // Default parameters
        
        return new SMC_LMB_Tracker(std::move(propagator), std::move(sensor_model), std::move(birth_model), survival_probability);
    }, "Creates a new SMC_LMB_Tracker with default model implementations");
    
    // Customizable factory function
    m.def("create_custom_smc_lmb_tracker", [](int particles_per_track, double initial_existence_prob, const Eigen::MatrixXd& covariance, double survival_probability) {
        auto propagator = std::make_unique<LinearPropagator>();
        auto sensor_model = std::make_unique<SimpleSensorModel>();
        auto birth_model = std::make_unique<AdaptiveBirthModel>(particles_per_track, initial_existence_prob, covariance);
        
        return new SMC_LMB_Tracker(std::move(propagator), std::move(sensor_model), std::move(birth_model), survival_probability);
    }, "Creates a new SMC_LMB_Tracker with custom birth model parameters");
}