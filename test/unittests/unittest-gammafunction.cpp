// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"

#include <iostream>

#include "gamma.h"

TEST_CASE("test the gamma function and incomplete gamma function.") {
    SECTION("double precision tests") {
        constexpr double TOL = 1e-8;

        double gamma;
        gamma = helpme::gammaComputer<double, -4>::value;
        REQUIRE(gamma == Approx(std::numeric_limits<double>::max()).margin(TOL));
        gamma = helpme::gammaComputer<double, -3>::value;
        REQUIRE(gamma == Approx(2.363271801).margin(TOL));
        gamma = helpme::gammaComputer<double, -2>::value;
        REQUIRE(gamma == Approx(std::numeric_limits<double>::max()).margin(TOL));
        gamma = helpme::gammaComputer<double, -1>::value;
        REQUIRE(gamma == Approx(-3.544907702).margin(TOL));
        gamma = helpme::gammaComputer<double, 0>::value;
        REQUIRE(gamma == Approx(std::numeric_limits<double>::max()).margin(TOL));
        gamma = helpme::gammaComputer<double, 1>::value;
        REQUIRE(gamma == Approx(1.772453851).margin(TOL));
        gamma = helpme::gammaComputer<double, 2>::value;
        REQUIRE(gamma == Approx(1.0).margin(TOL));
        gamma = helpme::gammaComputer<double, 3>::value;
        REQUIRE(gamma == Approx(8.862269255e-1).margin(TOL));
        gamma = helpme::gammaComputer<double, 4>::value;
        REQUIRE(gamma == Approx(1.0).margin(TOL));

        gamma = helpme::nonTemplateGammaComputer<double>(-4);
        REQUIRE(gamma == Approx(std::numeric_limits<double>::max()).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<double>(-3);
        REQUIRE(gamma == Approx(2.363271801).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<double>(-2);
        REQUIRE(gamma == Approx(std::numeric_limits<double>::max()).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<double>(-1);
        REQUIRE(gamma == Approx(-3.544907702).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<double>(0);
        REQUIRE(gamma == Approx(std::numeric_limits<double>::max()).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<double>(1);
        REQUIRE(gamma == Approx(1.772453851).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<double>(2);
        REQUIRE(gamma == Approx(1.0).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<double>(3);
        REQUIRE(gamma == Approx(8.862269255e-1).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<double>(4);
        REQUIRE(gamma == Approx(1.0).margin(TOL));

        REQUIRE(helpme::incompleteGammaComputer<double, -4>::compute(0.1) == Approx(4.162914579e1).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, -3>::compute(0.1) == Approx(1.680780146e1).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, -2>::compute(0.1) == Approx(7.225450222).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, -1>::compute(0.1) == Approx(3.401769337).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 0>::compute(0.1) == Approx(1.822923958).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 1>::compute(0.1) == Approx(1.160462485).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 2>::compute(0.1) == Approx(9.04837418e-1).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 3>::compute(0.1) == Approx(8.663659577e-1).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 4>::compute(0.1) == Approx(9.953211598e-1).margin(TOL));

        REQUIRE(helpme::incompleteGammaComputer<double, -4>::compute(3.0) == Approx(9.922940618e-4).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, -3>::compute(3.0) == Approx(1.870259849e-3).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, -2>::compute(3.0) == Approx(3.547308362e-3).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, -1>::compute(3.0) == Approx(6.776136002e-3).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 0>::compute(3.0) == Approx(1.304838109e-2).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 1>::compute(3.0) == Approx(2.535650932e-2).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 2>::compute(3.0) == Approx(4.978706837e-2).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 3>::compute(3.0) == Approx(9.891198663e-2).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 4>::compute(3.0) == Approx(1.991482735e-1).margin(TOL));

        REQUIRE(helpme::incompleteGammaComputer<double, 0>::compute(60.0) == Approx(1.43587e-28).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 0>::compute(10.0) == Approx(4.15697e-6).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 0>::compute(3.0) == Approx(0.0130484).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 0>::compute(0.0) ==
                Approx(std::numeric_limits<double>::max()).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 0>::compute(-3.0) == Approx(-9.93383).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 0>::compute(-10.0) == Approx(-2492.23).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<double, 0>::compute(-60.0) == Approx(-1.93618e24).margin(TOL));

        std::pair<double, double> pair;
        pair = helpme::incompleteGammaVirialComputer<double, -4>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(9.922940618e-4).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(3.547308362e-3).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<double, -3>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(1.870259849e-3).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(6.776136002e-3).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<double, -2>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(3.547308362e-3).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(1.304838109e-2).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<double, -1>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(6.776136002e-3).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(2.535650932e-2).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<double, 0>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(1.304838109e-2).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(4.978706837e-2).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<double, 1>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(2.535650932e-2).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(9.891198663e-2).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<double, 2>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(4.978706837e-2).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(1.991482735e-1).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<double, 3>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(9.891198663e-2).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(4.070691759e-1).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<double, 4>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(1.991482735e-1).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(8.463801623e-1).margin(TOL));
    }

    SECTION("single precision tests") {
        constexpr float TOL = 1e-6f;

        float gamma;
        gamma = helpme::gammaComputer<float, -4>::value;
        REQUIRE(gamma == Approx(std::numeric_limits<float>::max()).margin(TOL));
        gamma = helpme::gammaComputer<float, -3>::value;
        REQUIRE(gamma == Approx(2.363271801f).margin(TOL));
        gamma = helpme::gammaComputer<float, -2>::value;
        REQUIRE(gamma == Approx(std::numeric_limits<float>::max()).margin(TOL));
        gamma = helpme::gammaComputer<float, -1>::value;
        REQUIRE(gamma == Approx(-3.544907702f).margin(TOL));
        gamma = helpme::gammaComputer<float, 0>::value;
        REQUIRE(gamma == Approx(std::numeric_limits<float>::max()).margin(TOL));
        gamma = helpme::gammaComputer<float, 1>::value;
        REQUIRE(gamma == Approx(1.772453851f).margin(TOL));
        gamma = helpme::gammaComputer<float, 2>::value;
        REQUIRE(gamma == Approx(1.0f).margin(TOL));
        gamma = helpme::gammaComputer<float, 3>::value;
        REQUIRE(gamma == Approx(8.862269255e-1f).margin(TOL));
        gamma = helpme::gammaComputer<float, 4>::value;
        REQUIRE(gamma == Approx(1.0f).margin(TOL));

        gamma = helpme::nonTemplateGammaComputer<float>(-4);
        REQUIRE(gamma == Approx(std::numeric_limits<float>::max()).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<float>(-3);
        REQUIRE(gamma == Approx(2.363271801f).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<float>(-2);
        REQUIRE(gamma == Approx(std::numeric_limits<float>::max()).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<float>(-1);
        REQUIRE(gamma == Approx(-3.544907702f).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<float>(0);
        REQUIRE(gamma == Approx(std::numeric_limits<float>::max()).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<float>(1);
        REQUIRE(gamma == Approx(1.772453851f).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<float>(2);
        REQUIRE(gamma == Approx(1.0f).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<float>(3);
        REQUIRE(gamma == Approx(8.862269255e-1f).margin(TOL));
        gamma = helpme::nonTemplateGammaComputer<float>(4);
        REQUIRE(gamma == Approx(1.0f).margin(TOL));

        REQUIRE(helpme::incompleteGammaComputer<float, -4>::compute(0.1) == Approx(4.162914579e1f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, -3>::compute(0.1) == Approx(1.680780146e1f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, -2>::compute(0.1) == Approx(7.225450222f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, -1>::compute(0.1) == Approx(3.401769337f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 0>::compute(0.1) == Approx(1.822923958f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 1>::compute(0.1) == Approx(1.160462485f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 2>::compute(0.1) == Approx(9.04837418e-1f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 3>::compute(0.1) == Approx(8.663659577e-1f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 4>::compute(0.1) == Approx(9.953211598e-1f).margin(TOL));

        REQUIRE(helpme::incompleteGammaComputer<float, -4>::compute(3.0) == Approx(9.922940618e-4f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, -3>::compute(3.0) == Approx(1.870259849e-3f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, -2>::compute(3.0) == Approx(3.547308362e-3f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, -1>::compute(3.0) == Approx(6.776136002e-3f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 0>::compute(3.0) == Approx(1.304838109e-2f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 1>::compute(3.0) == Approx(2.535650932e-2f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 2>::compute(3.0) == Approx(4.978706837e-2f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 3>::compute(3.0) == Approx(9.891198663e-2f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 4>::compute(3.0) == Approx(1.991482735e-1f).margin(TOL));

        REQUIRE(helpme::incompleteGammaComputer<float, 0>::compute(60.0) == Approx(1.43587e-28f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 0>::compute(10.0) == Approx(4.15697e-6f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 0>::compute(3.0) == Approx(0.0130484f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 0>::compute(0.0) ==
                Approx(std::numeric_limits<float>::max()).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 0>::compute(-3.0) == Approx(-9.93383f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 0>::compute(-10.0) == Approx(-2492.23f).margin(TOL));
        REQUIRE(helpme::incompleteGammaComputer<float, 0>::compute(-60.0) == Approx(-1.93618e24f).margin(TOL));

        std::pair<float, float> pair;
        pair = helpme::incompleteGammaVirialComputer<float, -4>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(9.922940618e-4f).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(3.547308362e-3f).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<float, -3>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(1.870259849e-3f).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(6.776136002e-3f).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<float, -2>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(3.547308362e-3f).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(1.304838109e-2f).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<float, -1>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(6.776136002e-3f).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(2.535650932e-2f).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<float, 0>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(1.304838109e-2f).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(4.978706837e-2f).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<float, 1>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(2.535650932e-2f).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(9.891198663e-2f).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<float, 2>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(4.978706837e-2f).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(1.991482735e-1f).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<float, 3>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(9.891198663e-2f).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(4.070691759e-1f).margin(TOL));
        pair = helpme::incompleteGammaVirialComputer<float, 4>::compute(3.0);
        REQUIRE(std::get<0>(pair) == Approx(1.991482735e-1f).margin(TOL));
        REQUIRE(std::get<1>(pair) == Approx(8.463801623e-1f).margin(TOL));
    }
}
