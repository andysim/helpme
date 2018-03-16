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
#include "mpihelper.h"
#include "mpi_wrapper.h"
#include "helpme.h"

enum CalcType { E, EF, EFV };

template <typename Real>
std::tuple<Real, helpme::Matrix<Real>, helpme::Matrix<Real>> runTest(int nx, int ny, int nz, CalcType type) {
    float kappa = 0.3;
    int gridX = 32;
    int gridY = 32;
    int gridZ = 32;
    int splineOrder = 6;

    helpme::Matrix<Real> coords(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<Real> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    double scaleFactor = 332.0716;
    using PMEInstanceR = helpme::PMEInstance<Real>;
    auto pme = std::unique_ptr<PMEInstanceR>(new PMEInstanceR());
    Real parallelEnergy;
    helpme::Matrix<Real> nodeForces(6, 3);
    helpme::Matrix<Real> nodeVirial(6, 1);
    helpme::Matrix<Real> parallelForces(6, 3);
    helpme::Matrix<Real> parallelVirial(6, 1);

    bool serialRun = nx == 1 && ny == 1 && nz == 1;
    if (serialRun) {
        pme->setup(1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 1);
    } else {
        pme->setupParallel(1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 1, MPI_COMM_WORLD,
                           PMEInstanceR::NodeOrder::ZYX, nx, ny, nz);
    }
    pme->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceR::LatticeType::XAligned);
    Real nodeEnergy = 0;
    switch (type) {
        case E:
            nodeEnergy = pme->computeERec(0, charges, coords);
            break;
        case EF:
            nodeEnergy = pme->computeEFRec(0, charges, coords, nodeForces);
            break;
        case EFV:
            nodeEnergy = pme->computeEFVRec(0, charges, coords, nodeForces, nodeVirial);
            break;
        default:
            throw "Bad calculation type";
    }

    pme.reset();  // This is needed to avoid problems destroying the contained MPI communicators after MPI_Finalize.

    if (serialRun) {
        return std::make_tuple(nodeEnergy, std::move(nodeForces), std::move(nodeVirial));
    } else {
        // Only node 0 holds the results.   Use allreduce if all nodes need the info.
        helpme::MPITypes<Real> mpitype;
        MPI_Reduce(&nodeEnergy, &parallelEnergy, 1, mpitype.realType_, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeForces[0], parallelForces[0], 6 * 3, mpitype.realType_, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeVirial[0], parallelVirial[0], 6, mpitype.realType_, MPI_SUM, 0, MPI_COMM_WORLD);
        return std::make_tuple(parallelEnergy, std::move(parallelForces), std::move(parallelVirial));
    }
}

TEST_CASE("check consistency between serial and parallel results, for a toy Coulomb system.") {
    MPIHelper mpi;

    SECTION("Double precision tests") {
        double TOL = 1e-8;
        // Generate a serial benchmark first
        auto serialEFV = runTest<double>(1, 1, 1, EFV);
        double serialEnergy = std::get<0>(serialEFV);
        helpme::Matrix<double> serialForces = std::get<1>(serialEFV);
        helpme::Matrix<double> serialVirial = std::get<2>(serialEFV);
        SECTION("EFV tests") {
            SECTION("X partition") {
                auto xEFV = runTest<double>(2, 1, 1, EFV);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(xEFV);
                    auto forces = std::get<1>(xEFV);
                    auto virial = std::get<2>(xEFV);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                    REQUIRE(virial.almostEquals(serialVirial, TOL));
                }
            }
            SECTION("Y partition") {
                auto yEFV = runTest<double>(1, 2, 1, EFV);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(yEFV);
                    auto forces = std::get<1>(yEFV);
                    auto virial = std::get<2>(yEFV);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                    REQUIRE(virial.almostEquals(serialVirial, TOL));
                }
            }
            SECTION("Z partition") {
                auto zEFV = runTest<double>(1, 1, 2, EFV);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(zEFV);
                    auto forces = std::get<1>(zEFV);
                    auto virial = std::get<2>(zEFV);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                    REQUIRE(virial.almostEquals(serialVirial, TOL));
                }
            }
        }

        SECTION("EF tests") {
            SECTION("X partition") {
                auto xEF = runTest<double>(2, 1, 1, EF);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(xEF);
                    auto forces = std::get<1>(xEF);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                }
            }
            SECTION("Y partition") {
                auto yEF = runTest<double>(1, 2, 1, EF);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(yEF);
                    auto forces = std::get<1>(yEF);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                }
            }
            SECTION("Z partition") {
                auto zEF = runTest<double>(1, 1, 2, EF);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(zEF);
                    auto forces = std::get<1>(zEF);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                }
            }
        }

        SECTION("E tests") {
            SECTION("X partition") {
                auto xE = runTest<double>(2, 1, 1, E);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(xE);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                }
            }
            SECTION("Y partition") {
                auto yE = runTest<double>(1, 2, 1, E);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(yE);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                }
            }
            SECTION("Z partition") {
                auto zE = runTest<double>(1, 1, 2, E);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(zE);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                }
            }
        }
    }

    SECTION("Single precision tests") {
        double TOL = 1e-5;
        // Generate a serial benchmark first
        auto serialEFV = runTest<float>(1, 1, 1, EFV);
        float serialEnergy = std::get<0>(serialEFV);
        helpme::Matrix<float> serialForces = std::get<1>(serialEFV);
        helpme::Matrix<float> serialVirial = std::get<2>(serialEFV);
        SECTION("EFV tests") {
            SECTION("X partition") {
                auto xEFV = runTest<float>(2, 1, 1, EFV);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(xEFV);
                    auto forces = std::get<1>(xEFV);
                    auto virial = std::get<2>(xEFV);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                    REQUIRE(virial.almostEquals(serialVirial, TOL));
                }
            }
            SECTION("Y partition") {
                auto yEFV = runTest<float>(1, 2, 1, EFV);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(yEFV);
                    auto forces = std::get<1>(yEFV);
                    auto virial = std::get<2>(yEFV);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                    REQUIRE(virial.almostEquals(serialVirial, TOL));
                }
            }
            SECTION("Z partition") {
                auto zEFV = runTest<float>(1, 1, 2, EFV);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(zEFV);
                    auto forces = std::get<1>(zEFV);
                    auto virial = std::get<2>(zEFV);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                    REQUIRE(virial.almostEquals(serialVirial, TOL));
                }
            }
        }

        SECTION("EF tests") {
            SECTION("X partition") {
                auto xEF = runTest<float>(2, 1, 1, EF);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(xEF);
                    auto forces = std::get<1>(xEF);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                }
            }
            SECTION("Y partition") {
                auto yEF = runTest<float>(1, 2, 1, EF);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(yEF);
                    auto forces = std::get<1>(yEF);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                }
            }
            SECTION("Z partition") {
                auto zEF = runTest<float>(1, 1, 2, EF);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(zEF);
                    auto forces = std::get<1>(zEF);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                    REQUIRE(forces.almostEquals(serialForces, TOL));
                }
            }
        }

        SECTION("E tests") {
            SECTION("X partition") {
                auto xE = runTest<float>(2, 1, 1, E);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(xE);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                }
            }
            SECTION("Y partition") {
                auto yE = runTest<float>(1, 2, 1, E);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(yE);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                }
            }
            SECTION("Z partition") {
                auto zE = runTest<float>(1, 1, 2, E);
                if (mpi.myRank_ == 0) {
                    auto energy = std::get<0>(zE);
                    REQUIRE(energy == Approx(serialEnergy).margin(TOL));
                }
            }
        }
    }

    SECTION("Finalize MPI") { mpi.finalize(); }
}
