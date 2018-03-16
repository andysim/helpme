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

TEST_CASE("make sure lattice vectors are created correctly.") {
    SECTION("double precision tests") {
        auto pmeD = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
        pmeD->setup(1, 0.3, 6, 64, 64, 64, 1.9, 1);
        pmeD->setLatticeVectors(20, 22, 25, 70, 85, 100, PMEInstanceD::LatticeType::XAligned);
    }

    SECTION("single precision tests") {}
}
