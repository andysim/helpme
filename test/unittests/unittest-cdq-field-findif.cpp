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
#include <cstdlib>
#include <iomanip>
#include <iostream>


const char* valstr = std::getenv("HELPME_TESTS_NTHREADS");
int numThreads = valstr != NULL ? std::atoi(valstr) : 1;

TEST_CASE("check field by finite differences.") {
    std::cout << "Num Threads: " << numThreads << std::endl;

    // N.B. This test only passes for cubic test cases right now.  It's possible
    // the discrepancy is in the h0 term; more investigation needed!
    helpme::Matrix<double> coords(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<double> params({{-0.5196600000, 0.0000000000, 0.0000000000, 0.0755612194, 0.0484968397, 0.0000000000,
                                    -0.0534600659, 0.0000000000, 0.0000000000, 0.0049632262},
                                   {0.2598300000, 0.0320316208, 0.0000000000, 0.0184043956, -0.0002797814, 0.0000000000,
                                    -0.0137319752, 0.0181894766, 0.0000000000, 0.0140117566},
                                   {0.2598300000, -0.0320316208, 0.0000000000, 0.0184043956, -0.0002797814,
                                    0.0000000000, -0.0137319752, -0.0181894766, 0.0000000000, 0.0140117566},
                                   {-0.5196600000, 0.0000000000, 0.0000000000, 0.0755612194, 0.0484968397, 0.0000000000,
                                    -0.0534600659, 0.0000000000, 0.0000000000, 0.0049632262},
                                   {0.2598300000, 0.0320316208, 0.0000000000, 0.0184043956, -0.0002797814, 0.0000000000,
                                    -0.0137319752, 0.0181894766, 0.0000000000, 0.0140117566},
                                   {0.2598300000, -0.0320316208, 0.0000000000, 0.0184043956, -0.0002797814,
                                    0.0000000000, -0.0137319752, -0.0181894766, 0.0000000000, 0.0140117566}});
    double scaleFactor = 332.0637128;
    double kappa = 0.4;
    int gridPts = 64;

    helpme::PMEInstance<double> pme;
    pme.setup(1, kappa, 8, gridPts, gridPts, gridPts, scaleFactor, numThreads);
    pme.setLatticeVectors(25, 25, 25, 90, 90, 90, helpme::PMEInstance<double>::LatticeType::XAligned);
    helpme::Matrix<double> potential_an(6, 10);
    helpme::Matrix<double> potentialtmp(6, 1);
    double mutot[3] = {0, 0, 0};
    for (int atom = 0; atom < 6; ++atom) {
        mutot[0] += params[atom][1];
        mutot[1] += params[atom][2];
        mutot[2] += params[atom][3];
    }
    const double myPI = std::acos(-1.0);
    const double mySQRTPI = std::sqrt(myPI);

    pme.computePRec(2, params, coords, coords, 2, potential_an);

    double V = pme.cellVolume();
    double slfprefac = -scaleFactor * 4 * kappa * kappa * kappa / (3 * mySQRTPI);
    double h0prefac = scaleFactor * 4 * myPI / (3 * V);

    const double DELTA = 0.00001;
    const double TOL = 1e-8;
    for (int atom = 0; atom < 6; ++atom) {
        double *pC = coords[atom];
        for (int xyz = 0; xyz < 3; ++xyz) {
            // Numerically differentiate the potential w.r.t. geometry to get the field
            // Plus 1
            pC[xyz] += DELTA;
            potentialtmp.setZero();
            pme.computePRec(2, params, coords, coords, 0, potentialtmp);
            double PrecP1 = potentialtmp[atom][0];
            pC[xyz] -= DELTA;
            // Minus 1
            pC[xyz] -= DELTA;
            potentialtmp.setZero();
            pme.computePRec(2, params, coords, coords, 0, potentialtmp);
            double PrecM1 = potentialtmp[atom][0];
            pC[xyz] += DELTA;
            // Plus 2
            pC[xyz] += 2 * DELTA;
            potentialtmp.setZero();
            pme.computePRec(2, params, coords, coords, 0, potentialtmp);
            double PrecP2 = potentialtmp[atom][0];
            pC[xyz] -= 2 * DELTA;
            // Minus 2
            pC[xyz] -= 2 * DELTA;
            potentialtmp.setZero();
            pme.computePRec(2, params, coords, coords, 0, potentialtmp);
            double PrecM2 = potentialtmp[atom][0];
            pC[xyz] += 2 * DELTA;

            double field_fd = (PrecM2 + 8 * PrecP1 - 8 * PrecM1 - PrecP2) / (12 * DELTA);

            // We need to add some extra terms, per eqs. 31 and 32 of https://doi.org/10.1063/1.481216
            double mu = params[atom][xyz + 1];
            double h0term = h0prefac * mu;
            double slfterm = slfprefac * mu;
            double field_an = potential_an[atom][xyz + 1] + slfterm + h0term;
            // std::cout << field_fd - field_an << std::endl;
            REQUIRE(field_an == Approx(field_fd).margin(TOL));
        }
    }
}
