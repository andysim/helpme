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
    SECTION("potential tests up to field hessians") {
        helpme::Matrix<double> coords(
            {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
        helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
        double scaleFactor = 332.0637128;  // The factor used in Tinker
        double kappa = 0.3;
        int gridPts = 64;

        helpme::PMEInstance<double> pme;
        pme.setup(1, kappa, 6, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme.setLatticeVectors(20, 22, 25, 70, 85, 100, helpme::PMEInstance<double>::LatticeType::XAligned);
        double energy = pme.computeERec(0, charges, coords);
        helpme::Matrix<double> potential(6, 20);
        pme.computePRec(0, charges, coords, coords, 3, potential);
    }

    SECTION("check consistency of energy and potential") {
        std::vector<short> pairList;
        for (short i = 0; i < 6; ++i) {
            for (short j = 0; j < i; ++j) {
                pairList.push_back(i);
                pairList.push_back(j);
            }
        }
        auto pairs = helpme::Matrix<short>(pairList.data(), pairList.size() / 2, 2);
        helpme::Matrix<short> empty;
        helpme::Matrix<double> coords(
            {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
        helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
        double scaleFactor = 332.0637128;
        double kappa = 0.3;
        int gridPts = 64;

        helpme::PMEInstance<double> pme;
        pme.setup(1, kappa, 6, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme.setLatticeVectors(20, 22, 25, 70, 85, 100, helpme::PMEInstance<double>::LatticeType::XAligned);
        double Eall = pme.computeEAll(pairs, empty, 0, charges, coords);
        helpme::Matrix<double> potential(6, 1);
        pme.computePAtAtomicSites(charges, coords, potential, 9);
        double Epot = 0.5 * charges.dot(potential);
        REQUIRE(Epot == Approx(Eall).margin(1e-7));
    }

    SECTION("charge only potential at atomic sites") {
        // Place a few atoms near the boundaries so they get wrapped by the minimum image code
        helpme::Matrix<double> coords({
            {1.1, 1.2, 1.3},
            {10.1, 5.2, 3.3},
            {18.1, 18.1, 18.3},
            {0.1, 4.2, 2.3},
            {10.1, 3.2, 2.3},
            {26.1, 27.0, 20.3},
            {20.1, 28.2, 24.3},
            {5.1, 0.9, 3.3},
            {2.1, 2.1, 2.3},
        });
        helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
        double scaleFactor = 332.0637128;
        int gridPts = 96;
        // First, make sure the resulting potential is invariant to the attenuation parameter
        helpme::PMEInstance<double> pme3, pme4;
        pme3.setup(1, 0.4, 8, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme4.setup(1, 0.5, 8, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme3.setLatticeVectors(34, 33, 35, 80, 85, 100, helpme::PMEInstance<double>::LatticeType::XAligned);
        pme4.setLatticeVectors(34, 33, 35, 80, 85, 100, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> potential3(9, 1);
        helpme::Matrix<double> potential4(9, 1);
        pme3.computePAtAtomicSites(charges, coords, potential3, 15);
        pme4.computePAtAtomicSites(charges, coords, potential4, 15);
        REQUIRE(potential3.almostEquals(potential4));
        // Now make sure that it throws an error if the cutoff is too large
        REQUIRE_THROWS_WITH(pme3.computePAtAtomicSites(charges, coords, potential3, 25),
                            Catch::Contains("The cutoff used must be less than"));
    }

    SECTION("charge only potential at atomic sites, orthorhombic, compressed") {
        // Place a few atoms near the boundaries so they get wrapped by the minimum image code
        helpme::Matrix<double> coords({
            {1.1, 1.2, 1.3},
            {10.1, 5.2, 3.3},
            {18.1, 18.1, 18.3},
            {0.1, 4.2, 2.3},
            {10.1, 3.2, 2.3},
            {26.1, 27.0, 20.3},
            {20.1, 28.2, 24.3},
            {5.1, 0.9, 3.3},
            {2.1, 2.1, 2.3},
        });
        helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
        double scaleFactor = 332.0637128;
        int gridPts = 96;
        int kMax = 25;
        // First, make sure the resulting potential is invariant to the attenuation parameter
        helpme::PMEInstance<double> pme3, pme4;
        pme3.setupCompressed(1, 0.4, 8, gridPts, gridPts, gridPts, kMax, kMax, kMax, scaleFactor, 1);
        pme4.setupCompressed(1, 0.5, 8, gridPts, gridPts, gridPts, kMax, kMax, kMax, scaleFactor, 1);
        pme3.setLatticeVectors(34, 33, 35, 90, 90, 90, helpme::PMEInstance<double>::LatticeType::XAligned);
        pme4.setLatticeVectors(34, 33, 35, 90, 90, 90, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> potential3(9, 1);
        helpme::Matrix<double> potential4(9, 1);
        pme3.computePAtAtomicSites(charges, coords, potential3, 15);
        pme4.computePAtAtomicSites(charges, coords, potential4, 15);
        REQUIRE(potential3.almostEquals(potential4));
        // Now make sure that it throws an error if the cutoff is too large
        REQUIRE_THROWS_WITH(pme3.computePAtAtomicSites(charges, coords, potential3, 25),
                            Catch::Contains("The cutoff used must be less than"));
    }
}
