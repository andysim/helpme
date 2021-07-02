// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"

#include "helpme.h"
#include <cstdlib>
#include <iostream>

const char* valstr = std::getenv("HELPME_TESTS_NTHREADS");
int numThreads = valstr != NULL ? std::atoi(valstr) : 1;

TEST_CASE("test reciprocal space computations using only partial shells") {

    std::cout << "Num Threads: " << numThreads << std::endl;

    // Setup parameters and reference values.
    helpme::Matrix<double> coords(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<double> zeroChargeAndDipole({{0.00000, 0.0000000000, 0.0000, 0.0755612194},
                                                {0.00000, 0.0320316208, 0.0000, 0.0184043956},
                                                {0.00000, -0.0320316208, 0.0000, 0.0184043956},
                                                {0.00000, 0.0000000000, 0.0000, 0.0755612194},
                                                {0.00000, 0.0320316208, 0.0000, 0.0184043956},
                                                {0.00000, -0.0320316208, 0.0000, 0.0184043956}});
    helpme::Matrix<double> dipolesOnly({{0.0000000000, 0.0000000000, 0.0755612194},
                                        {0.0320316208, 0.0000000000, 0.0184043956},
                                        {-0.0320316208, 0.0000000000, 0.0184043956},
                                        {0.0000000000, 0.0000000000, 0.0755612194},
                                        {0.0320316208, 0.0000000000, 0.0184043956},
                                        {-0.0320316208, 0.0000000000, 0.0184043956}});
    short nfftx = 32;
    short nffty = 32;
    short nfftz = 32;
    double A = 20.0;
    double B = 20.0;
    double C = 20.0;

    SECTION("potential and field tests") {
        constexpr double TOL = 1e-7;
        double ccelec = 332.0716;
        int splineOrder = 5;

        helpme::Matrix<double> refPotentialAndField(6, 4);
        auto refpme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
        refpme->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, numThreads);
        refpme->setLatticeVectors(A, B, C, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        refpme->computePRec(1, zeroChargeAndDipole, coords, coords, 1, refPotentialAndField);

        SECTION("full potential and field from dipoles only") {
            helpme::Matrix<double> fullPotentialAndField(6, 4);
            auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
            pme->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, numThreads);
            pme->setLatticeVectors(A, B, C, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
            pme->computePRec(-1, dipolesOnly, coords, coords, 1, fullPotentialAndField);
            REQUIRE(fullPotentialAndField.almostEquals(refPotentialAndField, TOL));
        }

        SECTION("field only from dipoles only") {
            helpme::Matrix<double> fieldOnly(6, 3);
            auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
            pme->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, numThreads);
            pme->setLatticeVectors(A, B, C, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
            pme->computePRec(-1, dipolesOnly, coords, coords, -1, fieldOnly);
            for (int atom = 0; atom < 6; ++atom) {
                REQUIRE(fieldOnly[atom][0] == Approx(refPotentialAndField[atom][1]).margin(TOL));
                REQUIRE(fieldOnly[atom][1] == Approx(refPotentialAndField[atom][2]).margin(TOL));
                REQUIRE(fieldOnly[atom][2] == Approx(refPotentialAndField[atom][3]).margin(TOL));
            }
        }

        SECTION("full potential and field from full parameters with zero charges") {
            helpme::Matrix<double> fullPotentialAndField(6, 4);
            auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
            pme->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, numThreads);
            pme->setLatticeVectors(A, B, C, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
            pme->computePRec(1, zeroChargeAndDipole, coords, coords, 1, fullPotentialAndField);
            REQUIRE(fullPotentialAndField.almostEquals(refPotentialAndField, TOL));
        }
    }
}
