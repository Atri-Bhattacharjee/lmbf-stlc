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
#include "sgp4_propagator.h"
#include "in_orbit_sensor_model.h"
#include "assignment.h"
#include "metrics.h"

#include "two_body_propagator.h"

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
    
    pybind11::class_<Measurement>(m, "Measurement")
        .def(pybind11::init<>())
        .def_readwrite("timestamp_", &Measurement::timestamp_)
        .def_readwrite("value_", &Measurement::value_)
        .def_readwrite("covariance_", &Measurement::covariance_)
        .def_readwrite("sensor_id_", &Measurement::sensor_id_)
        .def_readwrite("sensor_state_", &Measurement::sensor_state_);
    
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
    
    // Bind abstract base interfaces
    pybind11::class_<IOrbitPropagator, std::shared_ptr<IOrbitPropagator>>(m, "IOrbitPropagator");
    pybind11::class_<ISensorModel, std::shared_ptr<ISensorModel>>(m, "ISensorModel");
    pybind11::class_<IBirthModel, std::shared_ptr<IBirthModel>>(m, "IBirthModel");
    
    // Bind concrete model implementations
    pybind11::class_<LinearPropagator, IOrbitPropagator, std::shared_ptr<LinearPropagator>>(m, "LinearPropagator")
        .def(pybind11::init<>(), "Default constructor for LinearPropagator")
        .def("propagate", &LinearPropagator::propagate);
    
    pybind11::class_<SimpleSensorModel, ISensorModel, std::shared_ptr<SimpleSensorModel>>(m, "SimpleSensorModel")
        .def(pybind11::init<>(), "Default constructor for SimpleSensorModel")
        .def("calculate_likelihood", &SimpleSensorModel::calculate_likelihood);
    
    pybind11::class_<AdaptiveBirthModel, IBirthModel, std::shared_ptr<AdaptiveBirthModel>>(m, "AdaptiveBirthModel")
        .def(pybind11::init<int, double, const Eigen::MatrixXd&>(),
             pybind11::arg("particles_per_track"),
             pybind11::arg("initial_existence_probability"),
             pybind11::arg("initial_covariance"),
             "Constructor for AdaptiveBirthModel")
        .def("generate_new_tracks", &AdaptiveBirthModel::generate_new_tracks);
    
    pybind11::class_<SGP4Propagator, IOrbitPropagator, std::shared_ptr<SGP4Propagator>>(m, "SGP4Propagator")
        .def(pybind11::init<const Eigen::MatrixXd&>(), pybind11::arg("process_noise_covariance"))
        .def("propagate", &SGP4Propagator::propagate);

    pybind11::class_<TwoBodyPropagator, IOrbitPropagator, std::shared_ptr<TwoBodyPropagator>>(m, "TwoBodyPropagator")
        .def(pybind11::init<const Eigen::MatrixXd&>(), pybind11::arg("process_noise_covariance"))
        .def("propagate", &TwoBodyPropagator::propagate);
    
    pybind11::class_<InOrbitSensorModel, ISensorModel, std::shared_ptr<InOrbitSensorModel>>(m, "InOrbitSensorModel")
        .def(pybind11::init<>(), "Default constructor for InOrbitSensorModel")
        .def("calculate_likelihood", &InOrbitSensorModel::calculate_likelihood);
    
    // Bind the main tracker class with direct constructor support
    pybind11::class_<SMC_LMB_Tracker, std::shared_ptr<SMC_LMB_Tracker>>(m, "SMC_LMB_Tracker")
        .def(pybind11::init<>(), "Default constructor for SMC_LMB_Tracker")
        .def(pybind11::init<std::shared_ptr<IOrbitPropagator>, std::shared_ptr<ISensorModel>, std::shared_ptr<IBirthModel>, double>(),
             pybind11::arg("propagator"),
             pybind11::arg("sensor_model"),
             pybind11::arg("birth_model"),
             pybind11::arg("survival_probability"),
             "Constructor for SMC_LMB_Tracker with model dependencies")
        .def("predict", &SMC_LMB_Tracker::predict, "Runs the predict step for a given time delta")
        .def("update", &SMC_LMB_Tracker::update, "Runs the update step with measurements")
        .def("get_tracks", &SMC_LMB_Tracker::get_tracks, "Gets the current list of tracks", pybind11::return_value_policy::reference_internal)
        .def("set_tracks", &SMC_LMB_Tracker::set_tracks, "Sets the initial list of tracks for the filter")
        .def("compute_association_likelihood", &SMC_LMB_Tracker::compute_association_likelihood, "Compute the association likelihood for a track and measurement");
    
    // Bind the Hypothesis struct
    pybind11::class_<Hypothesis>(m, "Hypothesis")
        .def(pybind11::init<>())
        .def_readwrite("associations", &Hypothesis::associations)
        .def_readwrite("weight", &Hypothesis::weight);

    // Bind the solve_assignment function
    m.def("solve_assignment", &solve_assignment, pybind11::arg("cost_matrix"), pybind11::arg("k_best"),
          "Solves the assignment problem and returns K-best hypotheses.");

    // Bind the calculate_ospa_distance function
    m.def("calculate_ospa_distance", &calculate_ospa_distance,
          "Calculates the OSPA distance between estimated tracks and ground truths",
          pybind11::arg("tracks"),
          pybind11::arg("ground_truths"),
          pybind11::arg("cutoff"));
}