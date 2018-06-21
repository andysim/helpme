// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"

#include <map>

#include "helpme.h"
#include <iomanip>

TEST_CASE("check potential (and derivatives thereof) code.") {
    int nAtoms = 6;
    helpme::Matrix<double> coords(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    double scaleFactor = 332.0637128;  // The factor used in Tinker
    double kappa = 0.3;
    int gridPts = 64;

    helpme::PMEInstance<double> pme;
    pme.setup(1, kappa, 6, gridPts, gridPts, gridPts, scaleFactor, 1);
    pme.setLatticeVectors(20, 22, 25, 70, 85, 100, helpme::PMEInstance<double>::LatticeType::XAligned);

    SECTION("potential tests") {
        double energy = pme.computeERec(0, charges, coords);
        std::cout << "E " << std::setprecision(16) << energy << std::endl;
        helpme::Matrix<double> potential(6, 20);
        pme.computePRec(0, charges, coords, coords, 3, potential);
        std::cout << potential << std::endl;
    }
}
