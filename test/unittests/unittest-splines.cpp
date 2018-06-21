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
#include "splines.h"

TEST_CASE("make sure B-Splines and their derivatives are created correctly.") {
    constexpr int order = 5;
    constexpr int deriv = 3;
    helpme::Matrix<double> refData({{5.56806667e-04, 1.31556773e-01, 5.83122173e-01, 2.76858107e-01, 7.90614000e-03},
                                    {-6.55066667e-03, -3.68264000e-01, -1.95904000e-01, 5.22802667e-01, 4.79160000e-02},
                                    {5.78000000e-02, 6.08800000e-01, -1.17320000e+00, 2.88800000e-01, 2.17800000e-01},
                                    {0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00}});
    helpme::Matrix<double> refMod6({1.0, 1.44, 4.0, 9.0, 4.0, 1.44});
    helpme::Matrix<double> refMod7(
        {1.0, 1.3076266068566, 2.8486134481222, 7.451136506592, 7.4511365065917, 2.848613448122, 1.3076266068566});
    helpme::Matrix<double> refMod8(
        {1.0, 1.2280943573293, 2.25, 5.3841505406298, 9.0, 5.3841505406299, 2.25, 1.2280943573293});

    SECTION("check spline differentiability") { REQUIRE_THROWS(helpme::BSpline<double>(1, 0.66, 6, 5)); }

    SECTION("double precision tests") {
        auto spline = helpme::BSpline<double>(1, 0.66, order, deriv);
        auto splineData = spline.splineData();

        // The number of rows must be deriv. lev. + 1 (4 in this case)
        REQUIRE(splineData.nRows() == deriv + 1);

        // The number of columns must be spline order.
        REQUIRE(splineData.nCols() == order);

        constexpr double TOL = 1e-8;
        REQUIRE(refData.almostEquals(splineData, TOL));

        // Make sure that the accessors work
        const auto* firstDerivative = spline[1];
        REQUIRE(firstDerivative[0] == Approx(-6.55066667e-03).margin(TOL));
        REQUIRE(firstDerivative[1] == Approx(-3.68264000e-01).margin(TOL));
        REQUIRE(firstDerivative[2] == Approx(-1.95904000e-01).margin(TOL));
        REQUIRE(firstDerivative[3] == Approx(5.22802667e-01).margin(TOL));
        REQUIRE(firstDerivative[4] == Approx(4.79160000e-02).margin(TOL));

        spline = helpme::BSpline<double>(0, 0.0, 4, 0);
        REQUIRE(refMod6.almostEquals(helpme::Matrix<double>(spline.invSplineModuli(6).data(), 6, 1), TOL));
        REQUIRE(refMod7.almostEquals(helpme::Matrix<double>(spline.invSplineModuli(7).data(), 7, 1), TOL));
        REQUIRE(refMod8.almostEquals(helpme::Matrix<double>(spline.invSplineModuli(8).data(), 8, 1), TOL));
    }

    SECTION("single precision tests") {
        auto spline = helpme::BSpline<float>(1, 0.66f, order, deriv);
        auto splineData = spline.splineData();

        // The number of rows must be deriv. lev. + 1 (4 in this case)
        REQUIRE(splineData.nRows() == deriv + 1);

        // The number of columns must be spline order.
        REQUIRE(splineData.nCols() == order);

        constexpr float TOL = 2e-6f;
        REQUIRE(refData.cast<float>().almostEquals(splineData, TOL));

        // Make sure that the accessors work
        const auto* firstDerivative = spline[1];
        REQUIRE(firstDerivative[0] == Approx(-6.55066667e-03).margin(TOL));
        REQUIRE(firstDerivative[1] == Approx(-3.68264000e-01).margin(TOL));
        REQUIRE(firstDerivative[2] == Approx(-1.95904000e-01).margin(TOL));
        REQUIRE(firstDerivative[3] == Approx(5.22802667e-01).margin(TOL));
        REQUIRE(firstDerivative[4] == Approx(4.79160000e-02).margin(TOL));

        spline = helpme::BSpline<float>(0, 0.0, 4, 0);
        REQUIRE(refMod6.cast<float>().almostEquals(helpme::Matrix<float>(spline.invSplineModuli(6).data(), 6, 1), TOL));
        REQUIRE(refMod7.cast<float>().almostEquals(helpme::Matrix<float>(spline.invSplineModuli(7).data(), 7, 1), TOL));
        REQUIRE(refMod8.cast<float>().almostEquals(helpme::Matrix<float>(spline.invSplineModuli(8).data(), 8, 1), TOL));
    }
}
