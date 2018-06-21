// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#include "helpme.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

namespace {
template <typename Real>
void declarePMEInstance(py::module& mod, std::string const& suffix) {
    using PME = helpme::PMEInstance<Real>;
    using Matrix = helpme::Matrix<Real>;

    py::class_<Matrix> mat(mod, ("Matrix" + suffix).c_str(), py::buffer_protocol());
    mat.def(py::init([](py::array_t<Real, py::array::c_style | py::array::forcecast> b) {
        /* Request a buffer descriptor from Python to construct a matrix from numpy arrays directly */
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<Real>::format())
            throw std::runtime_error("Incompatible format used to create Matrix py-side.");
        if (info.ndim != 2) throw std::runtime_error("Matrix object should have 2 dimensions.");
        return Matrix(static_cast<Real*>(info.ptr), info.shape[0], info.shape[1]);
    }));
    mat.def_buffer([](Matrix& m) -> py::buffer_info {
        return py::buffer_info(m[0],                                        /* Pointer to buffer */
                               sizeof(Real),                                /* Size of one scalar */
                               py::format_descriptor<Real>::format(),       /* Python struct-style format descriptor */
                               2,                                           /* Number of dimensions */
                               {m.nRows(), m.nCols()},                      /* Buffer dimensions */
                               {sizeof(Real) * m.nCols(), sizeof(Real) * 1} /* Strides (in bytes) for each index */
        );
    });

    py::class_<PME> pme(mod, ("PMEInstance" + suffix).c_str());
    pme.def(py::init<>(), "Construct PMEInstance object");
    pme.def("setup", &PME::setup, "Set up PMEInstance object for a serial run");
    pme.def("set_lattice_vectors", &PME::setLatticeVectors,
            "Set the lattice vectors for the unit cell: A, B, C, alpha, beta, gamma, Orietation.");
    pme.def("compute_E_rec", &PME::computeERec, py::arg("parameterAngMom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), "Computes the PME reciprocal space energy.");
    pme.def("compute_EF_rec", &PME::computeEFRec, py::arg("parameterAngMom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(),
            "Computes the PME reciprocal space energy and forces.");
    pme.def("compute_EFV_rec", &PME::computeEFVRec, py::arg("parameterAngMom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(), py::arg("virial").noconvert(),
            "Computes the PME reciprocal space energy, forces, and virial.");
    pme.def("compute_P_rec", &PME::computePRec, py::arg("parameterAngMom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("gridPoints").noconvert(), py::arg("derivativeLevel"),
            py::arg("potential").noconvert(),
            "Computes the PME reciprocal space potential and, optionally, its derivatives.");
    py::enum_<typename PME::LatticeType>(pme, "LatticeType")
        .value("ShapeMatrix", PME::LatticeType::ShapeMatrix)
        .value("XAligned", PME::LatticeType::XAligned);
}
}  // namespace

PYBIND11_MODULE(helpmelib, m) {
    m.doc() = R"pbdoc(
        helpme: an efficient library for particle mesh Ewald
                 ----------

                 .. currentmodule:: helpme 

                 .. autosummary::
                    :toctree: _generate

                 Molecule
    )pbdoc";

    declarePMEInstance<double>(m, "D");
    declarePMEInstance<float>(m, "F");

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif
}
