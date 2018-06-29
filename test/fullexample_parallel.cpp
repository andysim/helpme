// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include <mpi.h>
#include <cassert>
#include <stdlib.h>

#if BUILD_STANDALONE
#include "helpme_standalone.h"
#else
#include "helpme.h"
#endif

int main(int argc, char *argv[]) {
    MPI_Init(NULL, NULL);
    int numNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    const double tolerance = 1e-8;

    float kappa = 0.3;
    int gridX = 32;
    int gridY = 32;
    int gridZ = 32;
    int splineOrder = 6;

    helpme::Matrix<double> coords(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    double scaleFactor = 332.0716;
    helpme::Matrix<double> serialVirial(6, 1);
    helpme::Matrix<double> serialForces(6, 3);

    // Generate a serial benchmark first
    double energyS;
    if (myRank == 0) {
        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
        pme->setup(1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 1);
        pme->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        energyS = pme->computeEFVRec(0, charges, coords, serialForces, serialVirial);
        std::cout << "Serial results:" << std::endl;
        std::cout << "Total rec energy " << energyS << std::endl;
        std::cout << "Total forces" << std::endl << serialForces << std::endl;
        std::cout << "Total virial" << std::endl << serialVirial << std::endl;
    }

    // Now the parallel version
    auto pmeP = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
    if (argc == 4) {
        int nx = ::atoi(argv[1]);
        int ny = ::atoi(argv[2]);
        int nz = ::atoi(argv[3]);
        double parallelEnergy, nodeEnergy;
        helpme::Matrix<double> nodeForces(6, 3);
        helpme::Matrix<double> nodeVirial(6, 1);
        helpme::Matrix<double> parallelForces(6, 3);
        helpme::Matrix<double> parallelVirial(6, 1);

        nodeForces.setZero();
        nodeVirial.setZero();
        pmeP->setupParallel(1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 1, MPI_COMM_WORLD,
                            PMEInstanceD::NodeOrder::ZYX, nx, ny, nz);
        pmeP->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        nodeEnergy = pmeP->computeEFVRec(0, charges, coords, nodeForces, nodeVirial);
        MPI_Reduce(&nodeEnergy, &parallelEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeForces[0], parallelForces[0], 6 * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeVirial[0], parallelVirial[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myRank == 0) {
            std::cout << "Parallel results (nProcs = " << nx << ", " << ny << ", " << nz << "):" << std::endl;
            std::cout << "Total rec energy " << parallelEnergy << std::endl;
            std::cout << "Total forces " << std::endl << parallelForces << std::endl;
            std::cout << "Total virial " << std::endl << parallelVirial << std::endl;

            assert((std::abs(energyS - parallelEnergy) < tolerance));
            assert((serialForces.almostEquals(parallelForces, tolerance)));
            assert((serialVirial.almostEquals(parallelVirial, tolerance)));
        }
    } else {
        throw std::runtime_error(
            "This test should be run with exactly 3 arguments describing the number of X,Y and Z nodes.");
    }
    pmeP.reset();  // This ensures that the PME object cleans up its MPI data BEFORE MPI_Finalize is called;

    MPI_Finalize();

    return 0;
}
