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

const double TOL = 1e-8;

TEST_CASE("check small systems against reference values.") {
    // The reference values here are taken from Table 1 of https://doi.org/10.1063/1.481216
    helpme::Matrix<double> coords({{-0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}});

    int gridPts = 64;
    double kappa = 0.8;
    double kappa2 = kappa * kappa;
    const double myPI = std::acos(-1.0);
    double expterm = std::exp(-kappa2) / std::sqrt(myPI);
    double erfcterm = std::erfc(kappa);

    int splineOrder = 8;
    SECTION("two charges") {
        helpme::Matrix<double> forces(2, 3);
        helpme::Matrix<double> potential(2, 4);
        helpme::Matrix<double> charges({1.0, -1.0});
        helpme::PMEInstance<double> pme;
        pme.setup(1, kappa, splineOrder, gridPts, gridPts, gridPts, 1.0, 1);
        pme.setLatticeVectors(10, 10, 10, 90, 90, 90, helpme::PMEInstance<double>::LatticeType::XAligned);

        // Potential
        double realPotential = -erfcterm;
        double selfPotential = -2 * kappa / std::sqrt(myPI);
        pme.computePRec(0, charges, coords, coords, 1, potential);
        double recPotential = potential[0][0];
        REQUIRE(-1.0021255 == Approx(recPotential + realPotential + selfPotential).margin(TOL));

        // Field
        double realField = 2 * kappa * expterm + erfcterm;
        double recField = -potential[0][1];
        REQUIRE(0.9956865 == Approx(recField + realField).margin(TOL));

        // Energy
        double realEnergy = charges[0][0] * realPotential;
        double selfEnergy = charges[0][0] * selfPotential;
        double recEnergy = pme.computeERec(0, charges, coords);
        REQUIRE(-1.0021255 == Approx(recEnergy + realEnergy + selfEnergy).margin(TOL));

        // Force
        double efEnergy = pme.computeEFRec(0, charges, coords, forces);
        double realForce = charges[0][0] * realField;
        double recForce = forces[0][0];
        REQUIRE(efEnergy == recEnergy);
        REQUIRE(0.9956865 == Approx(recForce + realForce).margin(TOL));
    }

    SECTION("two dipoles") {
        helpme::Matrix<double> forces(2, 3);
        helpme::Matrix<double> potential(2, 4);
        helpme::Matrix<double> dipoles({{0.0, 1.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}});
        helpme::PMEInstance<double> pme;
        pme.setup(1, kappa, splineOrder, gridPts, gridPts, gridPts, 1.0, 1);
        pme.setLatticeVectors(10, 10, 10, 90, 90, 90, helpme::PMEInstance<double>::LatticeType::XAligned);

        // Potential
        pme.computePRec(1, dipoles, coords, coords, 1, potential);
        double realPotential = 2 * kappa * expterm + erfcterm;
        double recPotential = -potential[0][0];
        REQUIRE(0.9956865 == Approx(recPotential + realPotential).margin(TOL));

        // Field
        double realField = -4 * kappa * (1 + kappa2) * expterm - 2 * erfcterm;
        double recField = potential[0][1];
        double selfField = -4 * kappa * kappa2 / (3 * std::sqrt(myPI));
        REQUIRE(-2.0087525 == Approx(realField + selfField + recField).margin(TOL));

        // Energy
        double realEnergy = dipoles[0][1] * realField;
        double selfEnergy = dipoles[0][1] * selfField;
        double recEnergy = pme.computeERec(1, dipoles, coords);
        REQUIRE(-2.0087525 == Approx(realEnergy + selfEnergy + recEnergy).margin(TOL));

        // Force
        double efEnergy = pme.computeEFRec(1, dipoles, coords, forces);
        double realForce = 4 * kappa * (3 + 2 * (kappa2 + kappa2 * kappa2)) * expterm + 6 * erfcterm;
        double recForce = forces[0][0];
        REQUIRE(efEnergy == recEnergy);
        REQUIRE(5.9992460 == Approx(realForce + recForce).margin(TOL));
    }
}
