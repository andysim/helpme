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

TEST_CASE("Full run with a small toy system, comprising two water molecules with multipoles.") {
    short nfftx = 32;
    short nffty = 32;
    short nfftz = 32;
    double A = 20;
    double B = 20;
    double C = 20;
    short splineOrder = 8;

    // Setup parameters and reference values from Tinker
    SECTION("cubic box tests") {
        double refEnergy = 2.25604387;
        helpme::Matrix<double> refForcesD({{0.1748516222, 0.2316408016, -2.4442159588},
                                           {-0.1715164347, -0.1442931871, 0.9717034690},
                                           {-0.1006671795, -0.1827860988, 1.0007420119},
                                           {0.3822431439, 0.3526244916, -1.9482611604},
                                           {-0.1211148915, -0.1476302865, 1.2437491166},
                                           {-0.1637962948, -0.1095557375, 1.1762826647}});
        helpme::Matrix<double> refVirialD(
            {0.2569354171, 0.1915207854, 0.2382528442, 0.8416158549, 0.8629133269, -3.8258435201});
        helpme::Matrix<double> coordsD(
            {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
        helpme::Matrix<double> paramsD(
            {-0.5196600000, 0.2598300000, 0.2598300000, -0.5196600000, 0.2598300000, 0.2598300000});
        helpme::Matrix<double> inducedDipolesD({{0.0092251518, 0.0122228914, -0.0029141547},
                                                {0.0025446531, 0.0001505958, 0.0015839007},
                                                {0.0033022043, 0.0048054610, 0.0024044242},
                                                {0.0061471648, 0.0051142895, 0.0040991183},
                                                {0.0086468751, 0.0101671484, -0.0003683103},
                                                {0.0044661068, 0.0002475640, -0.0030269495}});
        SECTION("double precision tests") {
            constexpr double TOL = 1e-6;
            helpme::Matrix<double> forcesD(6, 3);
            helpme::Matrix<double> virialD(6, 1);
            forcesD.setZero();
            double ccelec = 332.0637128;

            auto pmeD = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
            pmeD->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, 1);
            pmeD->setLatticeVectors(A, B, C, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
            double energy = pmeD->computeEFVRecIsotropicInducedDipoles(
                0, paramsD, inducedDipolesD, PMEInstanceD::PolarizationType::Mutual, coordsD, forcesD, virialD);
            REQUIRE(refEnergy == Approx(energy).margin(TOL));
            REQUIRE(refForcesD.almostEquals(forcesD, TOL));
            REQUIRE(refVirialD.almostEquals(virialD, TOL));
        }

        SECTION("single precision tests") {
            constexpr double TOL = 5e-5;
            helpme::Matrix<float> forcesF(6, 3);
            helpme::Matrix<float> virialF(6, 1);
            forcesF.setZero();
            double ccelec = 332.0637128;

            auto pmeF = std::unique_ptr<PMEInstanceF>(new PMEInstanceF);
            pmeF->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, 1);
            pmeF->setLatticeVectors(A, B, C, 90, 90, 90, PMEInstanceF::LatticeType::XAligned);
            double energy = pmeF->computeEFVRecIsotropicInducedDipoles(
                0, paramsD.cast<float>(), inducedDipolesD.cast<float>(), PMEInstanceF::PolarizationType::Mutual,
                coordsD.cast<float>(), forcesF, virialF);
            REQUIRE(refEnergy == Approx(energy).margin(TOL));
            REQUIRE(refForcesD.cast<float>().almostEquals(forcesF, TOL));
            REQUIRE(refVirialD.cast<float>().almostEquals(virialF, TOL));
        }
    }

    // Setup parameters and reference values from Tinker
    SECTION("triclinic box tests") {
        double refEnergy = 2.24801447;
        helpme::Matrix<double> refForcesD({{0.1827116825, 0.2416590466, -2.4369129626},
                                           {-0.1758982487, -0.1502838247, 0.9668188606},
                                           {-0.1053416677, -0.1887448512, 0.9967539203},
                                           {0.3909518422, 0.3644305333, -1.9395365265},
                                           {-0.1250456696, -0.1525975423, 1.2403526688},
                                           {-0.1673779653, -0.1144633650, 1.1725240413}}

        );
        helpme::Matrix<double> refVirialD(
            {0.2419898394, 0.1888984123, 0.2036227585, 0.8066207181, 0.8162164984, -3.7864001207});
        helpme::Matrix<double> coordsD(
            {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
        helpme::Matrix<double> paramsD(
            {-0.5196600000, 0.2598300000, 0.2598300000, -0.5196600000, 0.2598300000, 0.2598300000});
        helpme::Matrix<double> inducedDipolesD({{0.0092586650, 0.0122623013, -0.0028753512},
                                                {0.0025698649, 0.0001725418, 0.0016116630},
                                                {0.0033222038, 0.0048282798, 0.0024184259},
                                                {0.0061837020, 0.0051629764, 0.0041447144},
                                                {0.0086711788, 0.0101863163, -0.0003469492},
                                                {0.0044797588, 0.0002630478, -0.0030137221}});
        SECTION("double precision tests") {
            constexpr double TOL = 1e-6;
            helpme::Matrix<double> forcesD(6, 3);
            helpme::Matrix<double> virialD(6, 1);
            forcesD.setZero();
            double ccelec = 332.0637128;

            auto pmeD = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
            pmeD->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, 1);
            pmeD->setLatticeVectors(A, B, C, 75, 80, 85, PMEInstanceD::LatticeType::XAligned);
            double energy = pmeD->computeEFVRecIsotropicInducedDipoles(
                0, paramsD, inducedDipolesD, PMEInstanceD::PolarizationType::Mutual, coordsD, forcesD, virialD);
            REQUIRE(refEnergy == Approx(energy).margin(TOL));
            REQUIRE(refForcesD.almostEquals(forcesD, TOL));
            REQUIRE(refVirialD.almostEquals(virialD, TOL));
        }

        SECTION("single precision tests") {
            constexpr double TOL = 5e-5;
            helpme::Matrix<float> forcesF(6, 3);
            helpme::Matrix<float> virialF(6, 1);
            forcesF.setZero();
            double ccelec = 332.0637128;

            auto pmeF = std::unique_ptr<PMEInstanceF>(new PMEInstanceF);
            pmeF->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, 1);
            pmeF->setLatticeVectors(A, B, C, 75, 80, 85, PMEInstanceF::LatticeType::XAligned);
            double energy = pmeF->computeEFVRecIsotropicInducedDipoles(
                0, paramsD.cast<float>(), inducedDipolesD.cast<float>(), PMEInstanceF::PolarizationType::Mutual,
                coordsD.cast<float>(), forcesF, virialF);
            REQUIRE(refEnergy == Approx(energy).margin(TOL));
            REQUIRE(refForcesD.cast<float>().almostEquals(forcesF, TOL));
            REQUIRE(refVirialD.cast<float>().almostEquals(virialF, TOL));
        }
    }
}
