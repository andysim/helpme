// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"

#include "gridsize.h"

TEST_CASE("test the grid size computing routines.") {
    REQUIRE(helpme::findGridSize(14, {4, 2}) == 16);
    REQUIRE(helpme::findGridSize(16, {4, 2}) == 16);
    REQUIRE(helpme::findGridSize(17, {4}) == 20);
    REQUIRE(helpme::findGridSize(41, {2}) == 42);
    REQUIRE(helpme::findGridSize(16, {2}) == 16);
    REQUIRE(helpme::findGridSize(123, {21}) == 126);
    REQUIRE(helpme::findGridSize(243, {8}) == 256);
    REQUIRE(helpme::findGridSize(243, {2}) == 250);
}
