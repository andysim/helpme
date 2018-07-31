// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"

#include "matrix.h"
#include "tensor_utils.h"

TEST_CASE("test the tensor handling routines.") {
    SECTION("resorting 3D tensors") {
        // These are 2x3x4 tensors
        helpme::Matrix<float> matrixABC(
            {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}});
        helpme::Matrix<float> matrixCBA(
            {{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}});
        helpme::Matrix<float> matrixACB(
            {{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}});

        SECTION("ABC->CBA") {
            helpme::Matrix<float> result(1, 24);
            helpme::permuteABCtoCBA(matrixABC[0], 2, 3, 4, result[0]);
            REQUIRE(result.almostEquals(matrixCBA));
        }

        SECTION("ABC->ACB") {
            helpme::Matrix<float> result(1, 24);
            helpme::permuteABCtoACB(matrixABC[0], 2, 3, 4, result[0]);
            REQUIRE(result.almostEquals(matrixACB));
        }
    }
}
