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
#include <iomanip>

const double DELTAR = 1e-4;
const double SQRTPI = std::sqrt(std::acos(-1.0));
const double TOL = 1e-6;

helpme::Matrix<double> computeRealField(double kappa, const helpme::Matrix<double>& coords_an,
                                        const helpme::Matrix<double>& charges_fd,
                                        const helpme::Matrix<double>& coords_fd) {
    REQUIRE(coords_fd.nRows() == 2 * coords_an.nRows());
    REQUIRE(coords_fd.nRows() == charges_fd.nRows());
    helpme::Matrix<double> field(coords_an.nRows(), 4);
    double kappa2 = kappa * kappa;
    for (int an = 0; an < coords_an.nRows(); ++an) {
        double V = 0;
        double Ex = 0;
        double Ey = 0;
        double Ez = 0;
        for (int fd = 0; fd < coords_fd.nRows(); ++fd) {
            auto deltaR = coords_fd.row(fd) - coords_an.row(an);
            double R2 = deltaR.dot(deltaR);
            double R = std::sqrt(R2);
            if (R > 1e-2) {
                double prefac = (2 * kappa * R * std::exp(-kappa2 * R2) / SQRTPI + std::erfc(kappa * R)) / (R * R2);
                double q = charges_fd[0][fd];
                V += q * std::erfc(kappa * R) / R;
                Ex += q * deltaR[0][0] * prefac;
                Ey += q * deltaR[0][1] * prefac;
                Ez += q * deltaR[0][2] * prefac;
            }
        }
        field[an][0] = V;
        field[an][1] = Ex;
        field[an][2] = Ey;
        field[an][3] = Ez;
    }
    return field;
}

TEST_CASE("dipole kappa sweep.") {
    helpme::Matrix<double> coords_an({{2.0, 2.5, 3.0}, {0.0, 0.0, 0.0}});
    const auto& crd = coords_an;
    helpme::Matrix<double> potential_an(2, 4);
    helpme::Matrix<double> potential_fd(2, 4);
    helpme::Matrix<double> coords_fd({{crd[0][0], crd[0][1], crd[0][2] + DELTAR},
                                      {crd[1][0], crd[1][1], crd[1][2] + DELTAR},
                                      {crd[0][0], crd[0][1], crd[0][2] - DELTAR},
                                      {crd[1][0], crd[1][1], crd[1][2] - DELTAR}});
    double mu1_z = 1.1;
    double mu2_z = 1.4;
    double q1_z = mu1_z / (2 * DELTAR);
    double q2_z = mu2_z / (2 * DELTAR);
    helpme::Matrix<double> params_an({{0.0, 0.0, 0.0, mu1_z}, {0.0, 0.0, 0.0, mu2_z}});
    helpme::Matrix<double> params_fd({q1_z, q2_z, -q1_z, -q2_z});
    double scaleFactor = 332.0637128;
    // Listing of interacing pairs in the finite difference model
    helpme::Matrix<short> pairList{{0, 1}, {0, 3}, {1, 2}, {2, 3}};

    SECTION("reciprocal space findif vs analytical") {
        // Check that the finite difference approximation to the dipole is correct
        int gridPts = 64;
        double kappa = 0.4;
        helpme::PMEInstance<double> pme;
        pme.setup(1, kappa, 8, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme.setLatticeVectors(22, 25, 27, 80, 90, 100, helpme::PMEInstance<double>::LatticeType::XAligned);
        pme.computePRec(0, params_fd, coords_fd, coords_an, 1, potential_fd);
        pme.computePRec(1, params_an, coords_an, coords_an, 1, potential_an);
        REQUIRE(potential_fd.almostEquals(potential_an, TOL));
    }

    SECTION("check invariance of potential and field w.r.t. kappa") {
        double gridPts = 128;
        auto potential3 = computeRealField(0.3, coords_an, params_fd, coords_fd) * scaleFactor;
        helpme::PMEInstance<double> pme3;
        auto selfPrefac3 = -scaleFactor * 4 * std::pow(0.3, 3) / (3 * SQRTPI);
        potential3 += params_an * selfPrefac3;
        pme3.setup(1, 0.3, 8, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme3.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        pme3.computePRec(1, params_an, coords_an, coords_an, 1, potential3);

        auto potential4 = computeRealField(0.4, coords_an, params_fd, coords_fd) * scaleFactor;
        helpme::PMEInstance<double> pme4;
        auto selfPrefac4 = -scaleFactor * 4 * std::pow(0.4, 3) / (3 * SQRTPI);
        potential4 += params_an * selfPrefac4;
        pme4.setup(1, 0.4, 9, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme4.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        pme4.computePRec(1, params_an, coords_an, coords_an, 1, potential4);
        REQUIRE(potential3.almostEquals(potential4, TOL));

        auto potential5 = computeRealField(0.5, coords_an, params_fd, coords_fd) * scaleFactor;
        helpme::PMEInstance<double> pme5;
        auto selfPrefac5 = -scaleFactor * 4 * std::pow(0.5, 3) / (3 * SQRTPI);
        potential5 += params_an * selfPrefac5;
        pme5.setup(1, 0.5, 10, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme5.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        pme5.computePRec(1, params_an, coords_an, coords_an, 1, potential5);
        REQUIRE(potential4.almostEquals(potential5, TOL));
    }

    SECTION("check invariance of E w.r.t. kappa") {
        double gridPts = 128;
        helpme::PMEInstance<double> pme3;
        pme3.setup(1, 0.3, 8, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme3.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        auto E3 = pme3.computeEDir(pairList, 0, params_fd, coords_fd);
        E3 -= scaleFactor * (mu1_z * mu1_z + mu2_z * mu2_z) * 2 * std::pow(0.3, 3) / (3 * SQRTPI);
        E3 += pme3.computeERec(1, params_an, coords_an);

        helpme::PMEInstance<double> pme4;
        pme4.setup(1, 0.4, 9, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme4.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        auto E4 = pme4.computeEDir(pairList, 0, params_fd, coords_fd);
        E4 -= scaleFactor * (mu1_z * mu1_z + mu2_z * mu2_z) * 2 * std::pow(0.4, 3) / (3 * SQRTPI);
        E4 += pme4.computeERec(1, params_an, coords_an);
        REQUIRE(E3 == Approx(E4).margin(TOL));

        helpme::PMEInstance<double> pme5;
        pme5.setup(1, 0.5, 10, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme5.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        auto E5 = pme5.computeEDir(pairList, 0, params_fd, coords_fd);
        E5 -= scaleFactor * (mu1_z * mu1_z + mu2_z * mu2_z) * 2 * std::pow(0.5, 3) / (3 * SQRTPI);
        E5 += pme5.computeERec(1, params_an, coords_an);
        REQUIRE(E4 == Approx(E5).margin(TOL));
    }

    SECTION("check invariance of EF w.r.t. kappa") {
        double gridPts = 128;
        helpme::PMEInstance<double> pme3;
        pme3.setup(1, 0.3, 8, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme3.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> forces3(2, 3), forces3_fd(4, 3);
        auto E3 = pme3.computeEFDir(pairList, 0, params_fd, coords_fd, forces3_fd);
        forces3[0][0] = forces3_fd[0][0] + forces3_fd[2][0];
        forces3[0][1] = forces3_fd[0][1] + forces3_fd[2][1];
        forces3[0][2] = forces3_fd[0][2] + forces3_fd[2][2];
        forces3[1][0] = forces3_fd[1][0] + forces3_fd[3][0];
        forces3[1][1] = forces3_fd[1][1] + forces3_fd[3][1];
        forces3[1][2] = forces3_fd[1][2] + forces3_fd[3][2];
        E3 -= scaleFactor * (mu1_z * mu1_z + mu2_z * mu2_z) * 2 * std::pow(0.3, 3) / (3 * SQRTPI);
        E3 += pme3.computeEFRec(1, params_an, coords_an, forces3);

        helpme::PMEInstance<double> pme4;
        pme4.setup(1, 0.4, 9, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme4.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> forces4(2, 3), forces4_fd(4, 3);
        auto E4 = pme4.computeEFDir(pairList, 0, params_fd, coords_fd, forces4_fd);
        forces4[0][0] = forces4_fd[0][0] + forces4_fd[2][0];
        forces4[0][1] = forces4_fd[0][1] + forces4_fd[2][1];
        forces4[0][2] = forces4_fd[0][2] + forces4_fd[2][2];
        forces4[1][0] = forces4_fd[1][0] + forces4_fd[3][0];
        forces4[1][1] = forces4_fd[1][1] + forces4_fd[3][1];
        forces4[1][2] = forces4_fd[1][2] + forces4_fd[3][2];
        E4 -= scaleFactor * (mu1_z * mu1_z + mu2_z * mu2_z) * 2 * std::pow(0.4, 3) / (3 * SQRTPI);
        E4 += pme4.computeEFRec(1, params_an, coords_an, forces4);
        REQUIRE(E3 == Approx(E4).margin(TOL));
        REQUIRE(forces3.almostEquals(forces4, TOL));

        helpme::PMEInstance<double> pme5;
        pme5.setup(1, 0.5, 10, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme5.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> forces5(2, 3), forces5_fd(4, 3);
        auto E5 = pme5.computeEFDir(pairList, 0, params_fd, coords_fd, forces5_fd);
        forces5[0][0] = forces5_fd[0][0] + forces5_fd[2][0];
        forces5[0][1] = forces5_fd[0][1] + forces5_fd[2][1];
        forces5[0][2] = forces5_fd[0][2] + forces5_fd[2][2];
        forces5[1][0] = forces5_fd[1][0] + forces5_fd[3][0];
        forces5[1][1] = forces5_fd[1][1] + forces5_fd[3][1];
        forces5[1][2] = forces5_fd[1][2] + forces5_fd[3][2];
        E5 -= scaleFactor * (mu1_z * mu1_z + mu2_z * mu2_z) * 2 * std::pow(0.5, 3) / (3 * SQRTPI);
        E5 += pme5.computeEFRec(1, params_an, coords_an, forces5);
        REQUIRE(E4 == Approx(E5).margin(TOL));
        REQUIRE(forces4.almostEquals(forces5, TOL));
    }

    SECTION("check invariance of EFV w.r.t. kappa") {
        double gridPts = 128;
        auto dR = coords_an.row(1) - coords_an.row(0);

        helpme::PMEInstance<double> pme3;
        pme3.setup(1, 0.3, 8, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme3.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> forces3(2, 3), forces3_fd(4, 3), virial3(6, 1);
        auto E3 = pme3.computeEFDir(pairList, 0, params_fd, coords_fd, forces3_fd);
        forces3[0][0] = forces3_fd[0][0] + forces3_fd[2][0];
        forces3[0][1] = forces3_fd[0][1] + forces3_fd[2][1];
        forces3[0][2] = forces3_fd[0][2] + forces3_fd[2][2];
        forces3[1][0] = forces3_fd[1][0] + forces3_fd[3][0];
        forces3[1][1] = forces3_fd[1][1] + forces3_fd[3][1];
        forces3[1][2] = forces3_fd[1][2] + forces3_fd[3][2];
        virial3[0][0] = -dR[0][0] * (forces3_fd[0][0] + forces3_fd[2][0]);
        virial3[1][0] = -0.5 * (dR[0][0] * (forces3_fd[0][1] + forces3_fd[2][1]) +
                                dR[0][1] * (forces3_fd[0][0] + forces3_fd[2][0]));
        virial3[2][0] = -dR[0][1] * (forces3_fd[0][1] + forces3_fd[2][1]);
        virial3[3][0] = -0.5 * (dR[0][0] * (forces3_fd[0][2] + forces3_fd[2][2]) +
                                dR[0][2] * (forces3_fd[0][0] + forces3_fd[2][0]));
        virial3[4][0] = -0.5 * (dR[0][1] * (forces3_fd[0][2] + forces3_fd[2][2]) +
                                dR[0][2] * (forces3_fd[0][1] + forces3_fd[2][1]));
        virial3[5][0] = -dR[0][2] * (forces3_fd[0][2] + forces3_fd[2][2]);
        E3 -= scaleFactor * (mu1_z * mu1_z + mu2_z * mu2_z) * 2 * std::pow(0.3, 3) / (3 * SQRTPI);
        E3 += pme3.computeEFVRec(1, params_an, coords_an, forces3, virial3);

        helpme::PMEInstance<double> pme4;
        pme4.setup(1, 0.4, 9, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme4.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> forces4(2, 3), forces4_fd(4, 3), virial4(6, 1);
        auto E4 = pme4.computeEFDir(pairList, 0, params_fd, coords_fd, forces4_fd);
        forces4[0][0] = forces4_fd[0][0] + forces4_fd[2][0];
        forces4[0][1] = forces4_fd[0][1] + forces4_fd[2][1];
        forces4[0][2] = forces4_fd[0][2] + forces4_fd[2][2];
        forces4[1][0] = forces4_fd[1][0] + forces4_fd[3][0];
        forces4[1][1] = forces4_fd[1][1] + forces4_fd[3][1];
        forces4[1][2] = forces4_fd[1][2] + forces4_fd[3][2];
        virial4[0][0] = -dR[0][0] * (forces4_fd[0][0] + forces4_fd[2][0]);
        virial4[1][0] = -0.5 * (dR[0][0] * (forces4_fd[0][1] + forces4_fd[2][1]) +
                                dR[0][1] * (forces4_fd[0][0] + forces4_fd[2][0]));
        virial4[2][0] = -dR[0][1] * (forces4_fd[0][1] + forces4_fd[2][1]);
        virial4[3][0] = -0.5 * (dR[0][0] * (forces4_fd[0][2] + forces4_fd[2][2]) +
                                dR[0][2] * (forces4_fd[0][0] + forces4_fd[2][0]));
        virial4[4][0] = -0.5 * (dR[0][1] * (forces4_fd[0][2] + forces4_fd[2][2]) +
                                dR[0][2] * (forces4_fd[0][1] + forces4_fd[2][1]));
        virial4[5][0] = -dR[0][2] * (forces4_fd[0][2] + forces4_fd[2][2]);
        E4 -= scaleFactor * (mu1_z * mu1_z + mu2_z * mu2_z) * 2 * std::pow(0.4, 3) / (3 * SQRTPI);
        E4 += pme4.computeEFVRec(1, params_an, coords_an, forces4, virial4);
        REQUIRE(E3 == Approx(E4).margin(TOL));
        REQUIRE(forces3.almostEquals(forces4, TOL));
        REQUIRE(virial3.almostEquals(virial4, TOL));

        helpme::PMEInstance<double> pme5;
        pme5.setup(1, 0.5, 10, gridPts, gridPts, gridPts, scaleFactor, 1);
        pme5.setLatticeVectors(35, 35, 35, 85, 90, 95, helpme::PMEInstance<double>::LatticeType::XAligned);
        helpme::Matrix<double> forces5(2, 3), forces5_fd(4, 3), virial5(6, 1);
        auto E5 = pme5.computeEFDir(pairList, 0, params_fd, coords_fd, forces5_fd);
        forces5[0][0] = forces5_fd[0][0] + forces5_fd[2][0];
        forces5[0][1] = forces5_fd[0][1] + forces5_fd[2][1];
        forces5[0][2] = forces5_fd[0][2] + forces5_fd[2][2];
        forces5[1][0] = forces5_fd[1][0] + forces5_fd[3][0];
        forces5[1][1] = forces5_fd[1][1] + forces5_fd[3][1];
        forces5[1][2] = forces5_fd[1][2] + forces5_fd[3][2];
        virial5[0][0] = -dR[0][0] * (forces5_fd[0][0] + forces5_fd[2][0]);
        virial5[1][0] = -0.5 * (dR[0][0] * (forces5_fd[0][1] + forces5_fd[2][1]) +
                                dR[0][1] * (forces5_fd[0][0] + forces5_fd[2][0]));
        virial5[2][0] = -dR[0][1] * (forces5_fd[0][1] + forces5_fd[2][1]);
        virial5[3][0] = -0.5 * (dR[0][0] * (forces5_fd[0][2] + forces5_fd[2][2]) +
                                dR[0][2] * (forces5_fd[0][0] + forces5_fd[2][0]));
        virial5[4][0] = -0.5 * (dR[0][1] * (forces5_fd[0][2] + forces5_fd[2][2]) +
                                dR[0][2] * (forces5_fd[0][1] + forces5_fd[2][1]));
        virial5[5][0] = -dR[0][2] * (forces5_fd[0][2] + forces5_fd[2][2]);
        E5 -= scaleFactor * (mu1_z * mu1_z + mu2_z * mu2_z) * 2 * std::pow(0.5, 3) / (3 * SQRTPI);
        E5 += pme5.computeEFVRec(1, params_an, coords_an, forces5, virial5);
        REQUIRE(E4 == Approx(E5).margin(TOL));
        REQUIRE(forces4.almostEquals(forces5, TOL));
        REQUIRE(virial4.almostEquals(virial5, TOL));
    }
}
