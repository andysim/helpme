// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"

#include "string_utils.h"

TEST_CASE("test the number formatting routines.") {
    SECTION("complex numbers") {
        REQUIRE(helpme::formatNumber(std::complex<double>(1.22222220123, 0.00000933131), 15, 8) ==
                "(     1.22222220,      0.00000933)");
        REQUIRE(helpme::formatNumber(std::complex<float>(1.22222220123, 0.00000933131), 15, 8) ==
                "(     1.22222221,      0.00000933)");
        REQUIRE(helpme::formatNumber(std::complex<double>(1.22222220123, 0.00000933131), 10, 4) ==
                "(    1.2222,     0.0000)");
        REQUIRE(helpme::formatNumber(std::complex<float>(1.22222220123, 0.00000933131), 10, 4) ==
                "(    1.2222,     0.0000)");
    }

    SECTION("complex numbers") {
        REQUIRE(helpme::formatNumber(1.22222220123, 15, 8) == "     1.22222220");
        REQUIRE(helpme::formatNumber(1.22222220123f, 15, 8) == "     1.22222221");
        REQUIRE(helpme::formatNumber(1.22222220123, 10, 4) == "    1.2222");
        REQUIRE(helpme::formatNumber(1.22222220123f, 10, 4) == "    1.2222");
    }
}
