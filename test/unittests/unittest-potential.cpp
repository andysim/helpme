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
#include <cstdlib>
#include <iomanip>
#include <iostream>

const char* valstr = std::getenv("HELPME_TESTS_NTHREADS");
int numThreads = valstr != NULL ? std::atoi(valstr) : 1;

TEST_CASE("check potential (and derivatives thereof) code.") {
    std::cout << "Num Threads: " << numThreads << std::endl;
    SECTION("potential tests up to field hessians") {
        helpme::Matrix<double> TinkerRef(
            {{1.24501399,  -0.73040252, -0.91060258, 7.74291464, 0.27864536, 0.17715017, 0.13197391,
              -0.48508899, -0.46749014, -0.34359531, 0.17964603, 0.05200381, 0.04070331, 0.27017151,
              -0.66350275, 0.11332203,  -0.68529486, 0.28235132, 0.30010701, -2.08588729},
             {7.90871301,  -1.21835044, -1.08663032, 6.25528685, -0.24193607, 0.27528916, -0.50627534,
              -0.47354697, -0.12993472, -2.02922474, 0.26386979, 0.03202289,  0.06988592, 0.31060404,
              -0.56229620, 0.03447673,  -0.60409337, 0.37234421, 0.27231342,  -1.46985315},
             {8.96068312,  -0.83968463, -1.33393043, 6.43054038, -0.51525544, 0.20741270, -0.56448834,
              0.13137382,  -0.16192001, -2.33100919, 0.26176837, 0.10282514,  0.04392296, 0.39015324,
              -0.61185212, 0.02765709,  -0.61320842, 0.21568565, 0.34995392,  -1.41597073},
             {-10.19688355, -1.38180652, -1.28477147, 6.26882489, 0.84460615, -0.32241125, 0.70303186,
              0.08961812,   0.12985346,  2.34775979,  0.42446511, 0.07035363, 0.07625631,  0.37039552,
              -0.58418367,  0.00977751,  -0.60459609, 0.31340956, 0.31518365, -1.42299671},
             {-3.56385866, -0.98881245, -1.11814950, 7.87520403, 0.36643512, -0.23103803, 0.06656097,
              0.06390394,  0.49858537,  0.66634200,  0.33273215, 0.10186727, 0.05755981,  0.33343905,
              -0.71500699, 0.07926764,  -0.69261134, 0.30604617, 0.34538277, -2.10762141},
             {-2.45053932, -1.17757056, -0.85203775, 7.46244617, 0.00828776, -0.28538417, 0.00478209,
              0.74333813,  0.40425202,  0.35607158,  0.34984119, 0.00799847, 0.06255179,  0.24689465,
              -0.60621509, 0.10479678,  -0.66985075, 0.29745955, 0.26293474, -2.02389087}});
        helpme::Matrix<double> coords(
            {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
        helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
        double scaleFactor = 332.0637128;  // The factor used in Tinker
        double kappa = 0.3;
        int gridPts = 64;

        helpme::PMEInstance<double> pme;
        pme.setup(1, kappa, 6, gridPts, gridPts, gridPts, scaleFactor, numThreads);
        pme.setLatticeVectors(20, 22, 25, 70, 85, 100, helpme::PMEInstance<double>::LatticeType::XAligned);
        double energy = pme.computeERec(0, charges, coords);
        helpme::Matrix<double> potential(6, 20);
        pme.computePRec(0, charges, coords, coords, 3, potential);
        REQUIRE(potential.almostEquals(TinkerRef));
    }

    SECTION("check consistency of energy and potential") {
        std::vector<short> pairList;
        for (short i = 0; i < 6; ++i) {
            for (short j = 0; j < i; ++j) {
                pairList.push_back(i);
                pairList.push_back(j);
            }
        }
        auto pairs = helpme::Matrix<short>(pairList.data(), pairList.size() / 2, 2);
        helpme::Matrix<short> empty;
        helpme::Matrix<double> coords(
            {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
        helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
        double scaleFactor = 332.0637128;
        double kappa = 0.3;
        int gridPts = 64;

        helpme::PMEInstance<double> pme;
        pme.setup(1, kappa, 6, gridPts, gridPts, gridPts, scaleFactor, numThreads);
        pme.setLatticeVectors(20, 22, 25, 70, 85, 100, helpme::PMEInstance<double>::LatticeType::XAligned);
        double Eall = pme.computeEAll(pairs, empty, 0, charges, coords);
        helpme::Matrix<double> potential(6, 1);
        pme.computePAtAtomicSites(charges, coords, potential, 9);
        double Epot = 0.5 * charges.dot(potential);
        REQUIRE(Epot == Approx(Eall).margin(1e-7));
    }

    SECTION("charge only potential at atomic sites") {
        // Place a few atoms near the boundaries so they get wrapped by the minimum image code
        helpme::Matrix<double> coords({
            {1.1, 1.2, 1.3},
            {10.1, 5.2, 3.3},
            {18.1, 18.1, 18.3},
            {0.1, 4.2, 2.3},
            {10.1, 3.2, 2.3},
            {26.1, 27.0, 20.3},
            {20.1, 28.2, 24.3},
            {5.1, 0.9, 3.3},
            {2.1, 2.1, 2.3},
        });
        helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
        double scaleFactor = 332.0637128;
        int gridPts = 96;
        // First, make sure the resulting potential is invariant to the attenuation parameter
        helpme::PMEInstance<double> pme3, pme4;
        pme3.setup(1, 0.4, 8, gridPts, gridPts, gridPts, scaleFactor, numThreads);
        pme4.setup(1, 0.5, 8, gridPts, gridPts, gridPts, scaleFactor, numThreads);
        pme3.setLatticeVectors(34, 33, 35, 80, 85, 100, helpme::PMEInstance<double>::LatticeType::XAligned);
        pme4.setLatticeVectors(34, 33, 35, 80, 85, 100, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> potential3(9, 1);
        helpme::Matrix<double> potential4(9, 1);
        pme3.computePAtAtomicSites(charges, coords, potential3, 15);
        pme4.computePAtAtomicSites(charges, coords, potential4, 15);
        REQUIRE(potential3.almostEquals(potential4));
        // Now make sure that it throws an error if the cutoff is too large
        REQUIRE_THROWS_WITH(pme3.computePAtAtomicSites(charges, coords, potential3, 25),
                            Catch::Contains("The cutoff used must be less than"));
    }

    SECTION("charge only potential at atomic sites, orthorhombic, compressed") {
        // Place a few atoms near the boundaries so they get wrapped by the minimum image code
        helpme::Matrix<double> coords({
            {1.1, 1.2, 1.3},
            {10.1, 5.2, 3.3},
            {18.1, 18.1, 18.3},
            {0.1, 4.2, 2.3},
            {10.1, 3.2, 2.3},
            {26.1, 27.0, 20.3},
            {20.1, 28.2, 24.3},
            {5.1, 0.9, 3.3},
            {2.1, 2.1, 2.3},
        });
        helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
        double scaleFactor = 332.0637128;
        int gridPts = 96;
        int kMax = 25;
        // First, make sure the resulting potential is invariant to the attenuation parameter
        helpme::PMEInstance<double> pme3, pme4;
        pme3.setupCompressed(1, 0.4, 8, gridPts, gridPts, gridPts, kMax, kMax, kMax, scaleFactor, numThreads);
        pme4.setupCompressed(1, 0.5, 8, gridPts, gridPts, gridPts, kMax, kMax, kMax, scaleFactor, numThreads);
        pme3.setLatticeVectors(34, 33, 35, 90, 90, 90, helpme::PMEInstance<double>::LatticeType::XAligned);
        pme4.setLatticeVectors(34, 33, 35, 90, 90, 90, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> potential3(9, 1);
        helpme::Matrix<double> potential4(9, 1);
        pme3.computePAtAtomicSites(charges, coords, potential3, 15);
        pme4.computePAtAtomicSites(charges, coords, potential4, 15);
        REQUIRE(potential3.almostEquals(potential4));
        // Now make sure that it throws an error if the cutoff is too large
        REQUIRE_THROWS_WITH(pme3.computePAtAtomicSites(charges, coords, potential3, 25),
                            Catch::Contains("The cutoff used must be less than"));
    }
}
