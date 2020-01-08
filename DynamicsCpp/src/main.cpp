#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <planning/dynamics.h>

// https://github.com/pybind/cmake_example
// pip install ./DynamicsCpp

namespace py = pybind11;

PYBIND11_PLUGIN(DynamicsCpp) {
    py::module mod("DynamicsCpp", "pybind11 basics module");

    py::class_<Dynamics, std::shared_ptr<Dynamics>> clsDynamics(mod, "Dynamics");
    clsDynamics.def(py::init());
    clsDynamics.def("fwd_dynamics", &Dynamics::fwd_dynamics); // one contact dynamics
    clsDynamics.def("fwd_dynamics_wo_contact", &Dynamics::fwd_dynamics_wo_contact);
    clsDynamics.def("fwd_dynamics_holonomic", &Dynamics::fwd_dynamics_holonomic);
    clsDynamics.def("change_weight", &Dynamics::change_weight);
    clsDynamics.def("change_param", &Dynamics::change_param);
    clsDynamics.def("get_fext", &Dynamics::get_fext);
    clsDynamics.def("change_holonomic_contact_param", &Dynamics::change_holonomic_contact_param);
    clsDynamics.def("change_mass_recal", &Dynamics::change_mass_recal);
    clsDynamics.def("fwd_dynamics_w_friction", &Dynamics::fwd_dynamics_w_friction);
    clsDynamics.def("fwd_dynamics_scaled_coordinate", &Dynamics::fwd_dynamics_scaled_coordinate);

    return mod.ptr();
}