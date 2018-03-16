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

enum CalcType { E, EF, EFV };

template <typename Real>
std::tuple<Real, helpme::Matrix<Real>, helpme::Matrix<Real>> dispersionKappaSweepHelper(Real kappa, CalcType type) {
    // A simple helper utility to set up a toy system and return properties, for a given attenuation parameter.
    int nAtoms = 6;
    helpme::Matrix<Real> coords(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<Real> c6s({24.392917702456206, 0.027455726292236412, 0.027455726292236412, 24.392917702456206,
                              0.027455726292236412, 0.027455726292236412});
    Real scaleFactor = -1;
    helpme::Matrix<Real> forces(nAtoms, 3);
    forces.setZero();
    helpme::Matrix<Real> virial(1, 6);
    virial.setZero();
    std::vector<short> includedList;
    std::vector<short> excludedList;
    std::map<size_t, bool> connectedMap;
    connectedMap[1 * nAtoms + 0] = false;
    connectedMap[2 * nAtoms + 0] = false;
    connectedMap[2 * nAtoms + 1] = false;
    connectedMap[4 * nAtoms + 3] = false;
    connectedMap[5 * nAtoms + 3] = false;
    connectedMap[5 * nAtoms + 4] = false;
    for (short i = 0; i < 6; ++i) {
        for (short j = 0; j < i; ++j) {
            if (connectedMap.count(i * nAtoms + j)) {
                excludedList.push_back(i);
                excludedList.push_back(j);
            } else {
                includedList.push_back(i);
                includedList.push_back(j);
            }
        }
    }
    auto includedPairs = helpme::Matrix<short>(includedList.data(), includedList.size() / 2, 2);
    auto excludedPairs = helpme::Matrix<short>(excludedList.data(), excludedList.size() / 2, 2);
    int gridPts = kappa < 0.4f ? 64 : 96;
    helpme::PMEInstance<Real> pme;
    pme.setup(6, kappa, 8, gridPts, gridPts, gridPts, scaleFactor, 1);
    pme.setLatticeVectors(20, 22, 25, 70, 85, 100, helpme::PMEInstance<Real>::LatticeType::XAligned);

    Real energy = 0;
    switch (type) {
        case E:
            energy = pme.computeEAll(includedPairs, excludedPairs, 0, c6s, coords);
            break;
        case EF:
            energy = pme.computeEFAll(includedPairs, excludedPairs, 0, c6s, coords, forces);
            break;
        case EFV:
            energy = pme.computeEFVAll(includedPairs, excludedPairs, 0, c6s, coords, forces, virial);
            break;
        default:
            throw "Bad calculation type";
    }

    return std::make_tuple(energy, std::move(forces), std::move(virial));
}

TEST_CASE(
    "check invariance of energy, force and virial, with respect to attenuation parameter, for a toy Coulomb system.") {
    SECTION("EFV routine tests") {
        double TOL = 1e-8;

        auto EFV25 = dispersionKappaSweepHelper<double>(0.25, EFV);
        auto energy25 = std::get<0>(EFV25);
        auto forces25 = std::get<1>(EFV25);
        auto virial25 = std::get<2>(EFV25);

        auto EFV35 = dispersionKappaSweepHelper<double>(0.35, EFV);
        auto energy35 = std::get<0>(EFV35);
        auto forces35 = std::get<1>(EFV35);
        auto virial35 = std::get<2>(EFV35);
        REQUIRE(energy25 == Approx(energy35).margin(TOL));
        REQUIRE(forces25.almostEquals(forces35, TOL));
        REQUIRE(virial25.almostEquals(virial35, TOL));

        auto EFV45 = dispersionKappaSweepHelper<double>(0.45, EFV);
        auto energy45 = std::get<0>(EFV45);
        auto forces45 = std::get<1>(EFV45);
        auto virial45 = std::get<2>(EFV45);
        REQUIRE(energy35 == Approx(energy45).margin(TOL));
        REQUIRE(forces35.almostEquals(forces45, TOL));
        REQUIRE(virial35.almostEquals(virial45, TOL));
    }

    SECTION("EF routine tests") {
        double TOL = 1e-8;

        auto EF25 = dispersionKappaSweepHelper<double>(0.25, EF);
        auto energy25 = std::get<0>(EF25);
        auto forces25 = std::get<1>(EF25);

        auto EF35 = dispersionKappaSweepHelper<double>(0.35, EF);
        auto energy35 = std::get<0>(EF35);
        auto forces35 = std::get<1>(EF35);
        REQUIRE(energy25 == Approx(energy35).margin(TOL));
        REQUIRE(forces25.almostEquals(forces35, TOL));

        auto EF45 = dispersionKappaSweepHelper<double>(0.45, EF);
        auto energy45 = std::get<0>(EF45);
        auto forces45 = std::get<1>(EF45);
        REQUIRE(energy35 == Approx(energy45).margin(TOL));
        REQUIRE(forces35.almostEquals(forces45, TOL));
    }

    SECTION("E routine tests") {
        double TOL = 1e-8;

        auto E25 = dispersionKappaSweepHelper<double>(0.25, E);
        auto energy25 = std::get<0>(E25);

        auto E35 = dispersionKappaSweepHelper<double>(0.35, E);
        auto energy35 = std::get<0>(E35);
        REQUIRE(energy25 == Approx(energy35).margin(TOL));

        auto E45 = dispersionKappaSweepHelper<double>(0.45, E);
        auto energy45 = std::get<0>(E45);
        REQUIRE(energy35 == Approx(energy45).margin(TOL));
    }
}
