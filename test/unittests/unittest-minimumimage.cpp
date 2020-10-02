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

const double TOL = 1e-10;

void check_vectors(std::array<double, 3> computed, std::array<double, 3> expected) {
    REQUIRE(expected[0] == Approx(computed[0]).margin(TOL));
    REQUIRE(expected[1] == Approx(computed[1]).margin(TOL));
    REQUIRE(expected[2] == Approx(computed[2]).margin(TOL));
}

TEST_CASE("test the minimum image convention code.") {
    /*
     *     0 1 2 3 4 5 6 7
     *     ----------------
     *  0 |                |
     *  1 |        0       |
     *  2 |                |
     *  3 |                |
     *  4 |  2     6     3 |
     *  5 |                |
     *  6 |                |
     *  7 |        1       |
     *     ----------------
     */
    helpme::Matrix<double> coords({{1.0, 4.0, 4.0},
                                   {7.0, 4.0, 4.0},
                                   {4.0, 1.0, 4.0},
                                   {4.0, 7.0, 4.0},
                                   {4.0, 4.0, 1.0},
                                   {4.0, 4.0, 7.0},
                                   {4.0, 4.0, 4.0}});
    int gridPts = 12;
    double kappa = 0.3;
    double boxLength = 8.0;
    helpme::PMEInstance<double> pme;
    pme.setup(1, kappa, 6, gridPts, gridPts, gridPts, 1.0, 1);
    pme.setLatticeVectors(boxLength, boxLength, boxLength, 90, 90, 90,
                          helpme::PMEInstance<double>::LatticeType::XAligned);
    std::array<double, 3> dR;
    // x direction
    dR = pme.minimumImageDeltaR(coords.row(0), coords.row(1));
    check_vectors(dR, {-2, 0, 0});
    dR = pme.minimumImageDeltaR(coords.row(0), coords.row(6));
    check_vectors(dR, {3, 0, 0});
    dR = pme.minimumImageDeltaR(coords.row(1), coords.row(6));
    check_vectors(dR, {-3, 0, 0});
    // y direction
    dR = pme.minimumImageDeltaR(coords.row(2), coords.row(3));
    check_vectors(dR, {0, -2, 0});
    dR = pme.minimumImageDeltaR(coords.row(2), coords.row(6));
    check_vectors(dR, {0, 3, 0});
    dR = pme.minimumImageDeltaR(coords.row(3), coords.row(6));
    check_vectors(dR, {0, -3, 0});
    // z direction
    dR = pme.minimumImageDeltaR(coords.row(4), coords.row(5));
    check_vectors(dR, {0, 0, -2});
    dR = pme.minimumImageDeltaR(coords.row(4), coords.row(6));
    check_vectors(dR, {0, 0, 3});
    dR = pme.minimumImageDeltaR(coords.row(5), coords.row(6));
    check_vectors(dR, {0, 0, -3});
}
