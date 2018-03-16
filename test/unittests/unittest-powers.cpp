// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"

#include "powers.h"

TEST_CASE("test the exponentiation template functions.") {
    SECTION("double precision tests") {
        REQUIRE(helpme::raiseToIntegerPower<double, 0>::pow(3.5) == 1.0);
        REQUIRE(helpme::raiseToIntegerPower<double, 1>::pow(3.5) == 3.5);
        REQUIRE(helpme::raiseToIntegerPower<double, 2>::pow(3.5) == 12.25);
        REQUIRE(helpme::raiseToIntegerPower<double, 3>::pow(3.5) == 42.875);

        REQUIRE(helpme::raiseNormToIntegerPower<double, -4>::compute(4.0) == 0.0625);
        REQUIRE(helpme::raiseNormToIntegerPower<double, -3>::compute(4.0) == 0.125);
        REQUIRE(helpme::raiseNormToIntegerPower<double, -2>::compute(4.0) == 0.25);
        REQUIRE(helpme::raiseNormToIntegerPower<double, -1>::compute(4.0) == 0.5);
        REQUIRE(helpme::raiseNormToIntegerPower<double, 0>::compute(4.0) == 1.0);
        REQUIRE(helpme::raiseNormToIntegerPower<double, 1>::compute(4.0) == 2.0);
        REQUIRE(helpme::raiseNormToIntegerPower<double, 2>::compute(4.0) == 4.0);
        REQUIRE(helpme::raiseNormToIntegerPower<double, 3>::compute(4.0) == 8.0);
        REQUIRE(helpme::raiseNormToIntegerPower<double, 4>::compute(4.0) == 16.0);
    }

    SECTION("single precision tests") {
        REQUIRE(helpme::raiseToIntegerPower<float, 0>::pow(3.5f) == 1.0f);
        REQUIRE(helpme::raiseToIntegerPower<float, 1>::pow(3.5f) == 3.5f);
        REQUIRE(helpme::raiseToIntegerPower<float, 2>::pow(3.5f) == 12.25f);
        REQUIRE(helpme::raiseToIntegerPower<float, 3>::pow(3.5f) == 42.875f);

        REQUIRE(helpme::raiseNormToIntegerPower<float, -4>::compute(4.0f) == 0.0625f);
        REQUIRE(helpme::raiseNormToIntegerPower<float, -3>::compute(4.0f) == 0.125f);
        REQUIRE(helpme::raiseNormToIntegerPower<float, -2>::compute(4.0f) == 0.25f);
        // REQUIRE(helpme::raiseNormToIntegerPower<float,-1>::compute(4.0f) == 0.5f);
        REQUIRE(helpme::raiseNormToIntegerPower<float, 0>::compute(4.0f) == 1.0f);
        REQUIRE(helpme::raiseNormToIntegerPower<float, 1>::compute(4.0f) == 2.0f);
        REQUIRE(helpme::raiseNormToIntegerPower<float, 2>::compute(4.0f) == 4.0f);
        REQUIRE(helpme::raiseNormToIntegerPower<float, 3>::compute(4.0f) == 8.0f);
        REQUIRE(helpme::raiseNormToIntegerPower<float, 4>::compute(4.0f) == 16.0f);
    }

    SECTION("integer tests") {
        REQUIRE(helpme::raiseToIntegerPower<int, 0>::pow(3) == 1);
        REQUIRE(helpme::raiseToIntegerPower<int, 1>::pow(3) == 3);
        REQUIRE(helpme::raiseToIntegerPower<int, 2>::pow(3) == 9);
        REQUIRE(helpme::raiseToIntegerPower<int, 3>::pow(3) == 27);
    }
}
