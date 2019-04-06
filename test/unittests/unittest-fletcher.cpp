// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"
#include "fletcher.h"

#include <iostream>

TEST_CASE("test the fletcher checksums.") {
    auto expr = "abcdefgh";
    SECTION("simple string test (will fail on big endian systems)") {
        REQUIRE(helpme::compute_fletcher_checksum16(expr, sizeof(expr)) == 1575);
        REQUIRE(helpme::compute_fletcher_checksum32(expr, sizeof(expr)) == 3957429649);
        REQUIRE(helpme::compute_fletcher_checksum64(expr, sizeof(expr)) == 3543817411021686982);
    }
    SECTION("recasting as int16 types (will fail on big endian systems)") {
        REQUIRE(helpme::compute_fletcher_checksum16(reinterpret_cast<const uint16_t*>(expr), 4) == 1575);
        REQUIRE(helpme::compute_fletcher_checksum32(reinterpret_cast<const uint16_t*>(expr), 4) == 3957429649);
        REQUIRE(helpme::compute_fletcher_checksum64(reinterpret_cast<const uint16_t*>(expr), 4) == 3543817411021686982);
    }
    SECTION("recasting as int32 types (will fail on big endian systems)") {
        REQUIRE(helpme::compute_fletcher_checksum16(reinterpret_cast<const uint32_t*>(expr), 2) == 1575);
        REQUIRE(helpme::compute_fletcher_checksum32(reinterpret_cast<const uint32_t*>(expr), 2) == 3957429649);
        REQUIRE(helpme::compute_fletcher_checksum64(reinterpret_cast<const uint32_t*>(expr), 2) == 3543817411021686982);
    }
    SECTION("recasting as int64 types (will fail on big endian systems)") {
        REQUIRE(helpme::compute_fletcher_checksum16(reinterpret_cast<const uint64_t*>(expr), 1) == 1575);
        REQUIRE(helpme::compute_fletcher_checksum32(reinterpret_cast<const uint64_t*>(expr), 1) == 3957429649);
        REQUIRE(helpme::compute_fletcher_checksum64(reinterpret_cast<const uint64_t*>(expr), 1) == 3543817411021686982);
    }
    SECTION("recasting as float types (will fail on big endian systems)") {
        REQUIRE(helpme::compute_fletcher_checksum16(reinterpret_cast<const float*>(expr), 2) == 1575);
        REQUIRE(helpme::compute_fletcher_checksum32(reinterpret_cast<const float*>(expr), 2) == 3957429649);
        REQUIRE(helpme::compute_fletcher_checksum64(reinterpret_cast<const float*>(expr), 2) == 3543817411021686982);
    }
    SECTION("recasting as double types (will fail on big endian systems)") {
        REQUIRE(helpme::compute_fletcher_checksum16(reinterpret_cast<const double*>(expr), 1) == 1575);
        REQUIRE(helpme::compute_fletcher_checksum32(reinterpret_cast<const double*>(expr), 1) == 3957429649);
        REQUIRE(helpme::compute_fletcher_checksum64(reinterpret_cast<const double*>(expr), 1) == 3543817411021686982);
    }
    SECTION("errors are thrown for tiny data") {
        REQUIRE_THROWS_WITH(helpme::compute_fletcher_checksum16(expr, 0), Catch::Contains("Not enough data"));
        REQUIRE_THROWS_WITH(helpme::compute_fletcher_checksum32(expr, 1), Catch::Contains("Not enough data"));
        REQUIRE_THROWS_WITH(helpme::compute_fletcher_checksum64(expr, 3), Catch::Contains("Not enough data"));
    }
}
