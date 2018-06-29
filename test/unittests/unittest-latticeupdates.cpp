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

TEST_CASE("check that updates of kappa and unit cell parameters give the correct behavior.") {
    constexpr double TOL = 1e-7;
    double ccelec = 332.0716;

    // Setup parameters and reference values.
    helpme::Matrix<double> coords(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    short nfftx = 20;
    short nffty = 21;
    short nfftz = 22;
    short splineOrder = 5;

    double refEnergy1 = 5.8537004;
    helpme::Matrix<double> refForces1({{-0.60004038, -0.74129836, 6.31176591},
                                       {0.50238424, 0.44175023, -2.53478813},
                                       {0.34430074, 0.54474056, -2.60670541},
                                       {-1.14160970, -1.04857552, 5.07874657},
                                       {0.40743265, 0.45709529, -3.21032689},
                                       {0.48655356, 0.34618192, -3.03887422}

    });
    helpme::Matrix<double> refVirial1({{0.61893621, 0.49018413, 0.54959991, 2.29084071, 2.35776919, -9.96248284}});

    double refEnergy2 = 5.855746495;
    helpme::Matrix<double> refForces2({{-0.60245163, -0.73942661, 6.31413552},
                                       {0.50396028, 0.44092114, -2.53631961},
                                       {0.34571162, 0.54380710, -2.60819278},
                                       {-1.14467175, -1.04727038, 5.08209801},
                                       {0.40871507, 0.45630372, -3.21165662},
                                       {0.48780357, 0.34545271, -3.04026873}});
    helpme::Matrix<double> refVirial2({{0.61199705, 0.49055936, 0.54134312, 2.28222850, 2.36819958, -9.94504929}});

    double refEnergy3 = 6.871736497;
    helpme::Matrix<double> refForces3({{-0.76684907, -0.94791216, 7.41216206},
                                       {0.61560181, 0.54001262, -2.95904725},
                                       {0.42875798, 0.68237489, -3.02853111},
                                       {-1.40351882, -1.29196455, 5.91724551},
                                       {0.52358846, 0.58722357, -3.77349856},
                                       {0.60098832, 0.43008731, -3.56873433}});
    helpme::Matrix<double> refVirial3({{0.69619831, 0.55213933, 0.60823304, 2.78138346, 2.86610280, -11.39571070}});

    double refEnergy4 = 6.8718135;
    helpme::Matrix<double> refForces4({{-0.76682663, -0.79604124, 7.42058381},
                                       {0.61559595, 0.45348406, -2.96384799},
                                       {0.42875295, 0.57298883, -3.03458962},
                                       {-1.40348529, -1.08508863, 5.92864929},
                                       {0.52356758, 0.49320229, -3.77867550},
                                       {0.60096417, 0.36124367, -3.57252549}});
    helpme::Matrix<double> refVirial4({{0.69622261, 0.55220016, 0.60784358, 2.78135060, 2.86647846, -11.39584386}});
    double refEnergy5 = 7.55899485;
    helpme::Matrix<double> refForces5({{-0.84350929, -0.87564537, 8.16264219},
                                       {0.67715554, 0.49883247, -3.26023278},
                                       {0.47162825, 0.63028771, -3.33804858},
                                       {-1.54383381, -1.19359749, 6.52151422},
                                       {0.57592433, 0.54252252, -4.15654305},
                                       {0.66106059, 0.39736803, -3.92977804}});
    helpme::Matrix<double> refVirial5({{0.76584487, 0.60742018, 0.66862794, 3.05948566, 3.15312630, -12.53542824}});

    double refEnergy6 = 7.558688228;
    helpme::Matrix<double> refForces6({{-0.84303319, -0.87557676, 8.16169899},
                                       {0.67691837, 0.49890558, -3.25971866},
                                       {0.47193007, 0.63032536, -3.33749220},
                                       {-1.54317624, -1.19339693, 6.52017613},
                                       {0.57607320, 0.54246197, -4.15560408},
                                       {0.66144590, 0.39732351, -3.92883941}});
    helpme::Matrix<double> refVirial6({{0.76545755, 0.60761239, 0.66897556, 3.05912581, 3.15274330, -12.53308757}});

    double refEnergy7 = 0.007626089169;
    helpme::Matrix<double> refForces7({{-0.00111343, -0.00115672, 0.00829002},
                                       {0.00081670, 0.00059903, -0.00320443},
                                       {0.00059187, 0.00079872, -0.00324183},
                                       {-0.00187292, -0.00145863, 0.00637289},
                                       {0.00076641, 0.00072244, -0.00423731},
                                       {0.00081164, 0.00049523, -0.00397890}});
    helpme::Matrix<double> refVirial7({{0.00068085, 0.00059011, 0.00057443, 0.00357792, 0.00368895, -0.01097979}});

    SECTION("EFV routines") {
        double energy;
        helpme::Matrix<double> forces(6, 3);
        helpme::Matrix<double> virial(1, 6);

        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
        pme->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, 1);

        // Start with random setup
        forces.setZero();
        virial.setZero();
        pme->setLatticeVectors(21, 22, 20, 93, 92, 90, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeEFVRec(0, charges, coords, forces, virial);
        REQUIRE(energy == Approx(refEnergy1).margin(TOL));
        REQUIRE(forces.almostEquals(refForces1));
        REQUIRE(virial.almostEquals(refVirial1));

        // Call update with same parameters, to make sure everything's the same
        forces.setZero();
        virial.setZero();
        pme->setLatticeVectors(21, 22, 20, 93, 92, 90, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeEFVRec(0, charges, coords, forces, virial);
        REQUIRE(energy == Approx(refEnergy1).margin(TOL));
        REQUIRE(forces.almostEquals(refForces1));
        REQUIRE(virial.almostEquals(refVirial1));

        // Now change the parameters, and make sure it updates correctly
        forces.setZero();
        virial.setZero();
        pme->setLatticeVectors(21.5, 22.2, 20.1, 94, 91, 91, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeEFVRec(0, charges, coords, forces, virial);
        REQUIRE(energy == Approx(refEnergy2).margin(TOL));
        REQUIRE(forces.almostEquals(refForces2));
        REQUIRE(virial.almostEquals(refVirial2));

        // Back to the original setup
        forces.setZero();
        virial.setZero();
        pme->setLatticeVectors(21, 22, 20, 93, 92, 90, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeEFVRec(0, charges, coords, forces, virial);
        REQUIRE(energy == Approx(refEnergy1).margin(TOL));
        REQUIRE(forces.almostEquals(refForces1));
        REQUIRE(virial.almostEquals(refVirial1));

        // Same, but new kappa value
        forces.setZero();
        virial.setZero();
        pme->setup(1, 0.32, splineOrder, nfftx, nffty, nfftz, ccelec, 1);
        energy = pme->computeEFVRec(0, charges, coords, forces, virial);
        REQUIRE(energy == Approx(refEnergy3).margin(TOL));
        REQUIRE(forces.almostEquals(refForces3));
        REQUIRE(virial.almostEquals(refVirial3));

        // Adjust the grid slightly
        forces.setZero();
        virial.setZero();
        pme->setup(1, 0.32, splineOrder, nfftx, nffty + 4, nfftz, ccelec, 1);
        energy = pme->computeEFVRec(0, charges, coords, forces, virial);
        REQUIRE(energy == Approx(refEnergy4).margin(TOL));
        REQUIRE(forces.almostEquals(refForces4));
        REQUIRE(virial.almostEquals(refVirial4));

        // Adjust the scale factor slightly
        forces.setZero();
        virial.setZero();
        pme->setup(1, 0.32, splineOrder, nfftx, nffty + 4, nfftz, 1.1 * ccelec, 1);
        energy = pme->computeEFVRec(0, charges, coords, forces, virial);
        REQUIRE(energy == Approx(refEnergy5).margin(TOL));
        REQUIRE(forces.almostEquals(refForces5));
        REQUIRE(virial.almostEquals(refVirial5));

        // Adjust the scale factor slightly
        forces.setZero();
        virial.setZero();
        pme->setup(1, 0.32, splineOrder + 1, nfftx, nffty + 4, nfftz, 1.1 * ccelec, 1);
        energy = pme->computeEFVRec(0, charges, coords, forces, virial);
        REQUIRE(energy == Approx(refEnergy6).margin(TOL));
        REQUIRE(forces.almostEquals(refForces6));
        REQUIRE(virial.almostEquals(refVirial6));

        // Change the physics from coulomb to some weird disperion
        forces.setZero();
        virial.setZero();
        pme->setup(6, 0.32, splineOrder + 1, nfftx, nffty + 4, nfftz, 1.1 * ccelec, 1);
        energy = pme->computeEFVRec(0, charges, coords, forces, virial);
        REQUIRE(energy == Approx(refEnergy7).margin(TOL));
        REQUIRE(forces.almostEquals(refForces7));
        REQUIRE(virial.almostEquals(refVirial7));
    }

    SECTION("EF routines") {
        double energy;
        helpme::Matrix<double> forces(6, 3);

        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
        pme->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, 1);

        // Start with random setup
        forces.setZero();
        pme->setLatticeVectors(21, 22, 20, 93, 92, 90, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeEFRec(0, charges, coords, forces);
        REQUIRE(energy == Approx(refEnergy1).margin(TOL));
        REQUIRE(forces.almostEquals(refForces1));

        // Call update with same parameters, to make sure everything's the same
        forces.setZero();
        pme->setLatticeVectors(21, 22, 20, 93, 92, 90, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeEFRec(0, charges, coords, forces);
        REQUIRE(energy == Approx(refEnergy1).margin(TOL));
        REQUIRE(forces.almostEquals(refForces1));

        // Now change the parameters, and make sure it updates correctly
        forces.setZero();
        pme->setLatticeVectors(21.5, 22.2, 20.1, 94, 91, 91, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeEFRec(0, charges, coords, forces);
        REQUIRE(energy == Approx(refEnergy2).margin(TOL));
        REQUIRE(forces.almostEquals(refForces2));

        // Back to the original setup
        forces.setZero();
        pme->setLatticeVectors(21, 22, 20, 93, 92, 90, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeEFRec(0, charges, coords, forces);
        REQUIRE(energy == Approx(refEnergy1).margin(TOL));
        REQUIRE(forces.almostEquals(refForces1));

        // Same, but new kappa value
        forces.setZero();
        pme->setup(1, 0.32, splineOrder, nfftx, nffty, nfftz, ccelec, 1);
        energy = pme->computeEFRec(0, charges, coords, forces);
        REQUIRE(energy == Approx(refEnergy3).margin(TOL));
        REQUIRE(forces.almostEquals(refForces3));

        // Adjust the grid slightly
        forces.setZero();
        pme->setup(1, 0.32, splineOrder, nfftx, nffty + 4, nfftz, ccelec, 1);
        energy = pme->computeEFRec(0, charges, coords, forces);
        REQUIRE(energy == Approx(refEnergy4).margin(TOL));
        REQUIRE(forces.almostEquals(refForces4));

        // Adjust the scale factor slightly
        forces.setZero();
        pme->setup(1, 0.32, splineOrder, nfftx, nffty + 4, nfftz, 1.1 * ccelec, 1);
        energy = pme->computeEFRec(0, charges, coords, forces);
        REQUIRE(energy == Approx(refEnergy5).margin(TOL));
        REQUIRE(forces.almostEquals(refForces5));

        // Adjust the scale factor slightly
        forces.setZero();
        pme->setup(1, 0.32, splineOrder + 1, nfftx, nffty + 4, nfftz, 1.1 * ccelec, 1);
        energy = pme->computeEFRec(0, charges, coords, forces);
        REQUIRE(energy == Approx(refEnergy6).margin(TOL));
        REQUIRE(forces.almostEquals(refForces6));

        // Change the physics from coulomb to some weird disperion
        forces.setZero();
        pme->setup(6, 0.32, splineOrder + 1, nfftx, nffty + 4, nfftz, 1.1 * ccelec, 1);
        energy = pme->computeEFRec(0, charges, coords, forces);
        REQUIRE(energy == Approx(refEnergy7).margin(TOL));
        REQUIRE(forces.almostEquals(refForces7));
    }

    SECTION("E routines") {
        double energy;

        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
        pme->setup(1, 0.3, splineOrder, nfftx, nffty, nfftz, ccelec, 1);

        // Start with random setup
        pme->setLatticeVectors(21, 22, 20, 93, 92, 90, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeERec(0, charges, coords);
        REQUIRE(energy == Approx(refEnergy1).margin(TOL));

        // Call update with same parameters, to make sure everything's the same
        pme->setLatticeVectors(21, 22, 20, 93, 92, 90, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeERec(0, charges, coords);
        REQUIRE(energy == Approx(refEnergy1).margin(TOL));

        // Now change the parameters, and make sure it updates correctly
        pme->setLatticeVectors(21.5, 22.2, 20.1, 94, 91, 91, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeERec(0, charges, coords);
        REQUIRE(energy == Approx(refEnergy2).margin(TOL));

        // Back to the original setup
        pme->setLatticeVectors(21, 22, 20, 93, 92, 90, PMEInstanceD::LatticeType::XAligned);
        energy = pme->computeERec(0, charges, coords);
        REQUIRE(energy == Approx(refEnergy1).margin(TOL));

        // Same, but new kappa value
        pme->setup(1, 0.32, splineOrder, nfftx, nffty, nfftz, ccelec, 1);
        energy = pme->computeERec(0, charges, coords);
        REQUIRE(energy == Approx(refEnergy3).margin(TOL));

        // Adjust the grid slightly
        pme->setup(1, 0.32, splineOrder, nfftx, nffty + 4, nfftz, ccelec, 1);
        energy = pme->computeERec(0, charges, coords);
        REQUIRE(energy == Approx(refEnergy4).margin(TOL));

        // Adjust the scale factor slightly
        pme->setup(1, 0.32, splineOrder, nfftx, nffty + 4, nfftz, 1.1 * ccelec, 1);
        energy = pme->computeERec(0, charges, coords);
        REQUIRE(energy == Approx(refEnergy5).margin(TOL));

        // Adjust the scale factor slightly
        pme->setup(1, 0.32, splineOrder + 1, nfftx, nffty + 4, nfftz, 1.1 * ccelec, 1);
        energy = pme->computeERec(0, charges, coords);
        REQUIRE(energy == Approx(refEnergy6).margin(TOL));

        // Change the physics from coulomb to some weird disperion
        pme->setup(6, 0.32, splineOrder + 1, nfftx, nffty + 4, nfftz, 1.1 * ccelec, 1);
        energy = pme->computeERec(0, charges, coords);
        REQUIRE(energy == Approx(refEnergy7).margin(TOL));
    }
}
