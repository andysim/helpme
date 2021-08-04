// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "helpme.h"
#include "print_results.h"

int main(int argc, char *argv[]) {
    int nx;
    int ny;
    int nz;
    int numThreads;
    if (argc == 5) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        nz = atoi(argv[3]);
        numThreads = atoi(argv[4]);
    } else {
        printf("This test should be run with exactly 4 arguments describing the number of X,Y and Z nodes and number of threads.");
        exit(1);
    }

    MPI_Init(NULL, NULL);
    int numNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    double tolerance = 1e-8;
    float kappa = 0.3;
    int gridX = 32;
    int gridY = 32;
    int gridZ = 32;
    int splineOrder = 6;

    /*
     * Instantiate double precision PME object
     */
    double coords[18] = {
          2.0, 2.0, 2.0,
          2.5, 2.0, 3.0,
          1.5, 2.0, 3.0,
          0.0, 0.0, 0.0,
          0.5, 0.0, 1.0,
         -0.5, 0.0, 1.0
    };
    double charges[6] = {-0.834, 0.417, 0.417, -0.834, 0.417, 0.417};
    double scaleFactor = 332.0716;

    double energyS = 0;
    double forcesS[18] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    double virialS[6] = {0, 0, 0, 0, 0, 0};

    if (myRank == 0) {
        printf("Num Threads: %d\n", numThreads);
        // Generate a serial benchmark first
        PMEInstance *pmeS = helpme_createD();
        helpme_setupD(pmeS, 1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, numThreads);
        helpme_set_lattice_vectorsD(pmeS, 20, 20, 20, 90, 90, 90, XAligned);
        // Compute the energy, forces, and virial
        energyS = helpme_compute_EFV_recD(pmeS, 6, 0, &charges[0], &coords[0], &forcesS[0], &virialS[0]);
        print_resultsD(6, "Serial Results:", energyS, forcesS, virialS);
        helpme_destroyD(pmeS);
    }

    // Now the parallel version
    PMEInstance *pmeP = helpme_createD();
    double nodeEnergy, parallelEnergy;
    double nodeForces[18] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    double nodeVirial[6] = {0, 0, 0, 0, 0, 0};
    double parallelForces[18] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    double parallelVirial[6] = {0, 0, 0, 0, 0, 0};
    helpme_setup_parallelD(pmeP, 1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 1, MPI_COMM_WORLD, ZYX, nx, ny, nz);
    helpme_set_lattice_vectorsD(pmeP, 20, 20, 20, 90, 90, 90, XAligned);
    nodeEnergy = helpme_compute_EFV_recD(pmeP, 6, 0, &charges[0], &coords[0], &nodeForces[0], &nodeVirial[0]);
    MPI_Reduce(&nodeEnergy, &parallelEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nodeForces[0], &parallelForces[0], 6 * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nodeVirial[0], &parallelVirial[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    char nodesStr[80];
    sprintf(&nodesStr[0], "Parallel results (nProcs = %d, %d, %d):", nx, ny, nz);
    if (myRank == 0) {
        print_resultsD(6, &nodesStr[0], parallelEnergy, parallelForces, parallelVirial);

        assert_close(1, &energyS, (void*) &parallelEnergy, tolerance, sizeof(double),  __FILE__, __LINE__);
        assert_close(18, forcesS, (void*) parallelForces, tolerance, sizeof(double),  __FILE__, __LINE__);
        assert_close(6, virialS, (void*) parallelVirial, tolerance, sizeof(double),  __FILE__, __LINE__);
    }
    // Now the compressed version
    memset(nodeForces, 0, 18 * sizeof(double));
    memset(nodeVirial, 0, 6 * sizeof(double));
    helpme_setup_compressed_parallelD(pmeP, 1, kappa, splineOrder, gridX, gridY, gridZ, 9, 9, 9, scaleFactor, 1, MPI_COMM_WORLD, ZYX, nx, ny, nz);
    helpme_set_lattice_vectorsD(pmeP, 20, 20, 20, 90, 90, 90, XAligned);
    nodeEnergy = helpme_compute_EFV_recD(pmeP, 6, 0, &charges[0], &coords[0], &nodeForces[0], &nodeVirial[0]);
    MPI_Reduce(&nodeEnergy, &parallelEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nodeForces[0], &parallelForces[0], 6 * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nodeVirial[0], &parallelVirial[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    sprintf(&nodesStr[0], "Parallel results (nProcs = %d, %d, %d):", nx, ny, nz);
    if (myRank == 0) {
        printf("\nCompressed\n");
        print_resultsD(6, &nodesStr[0], parallelEnergy, parallelForces, parallelVirial);

        assert_close(1, &energyS, (void*) &parallelEnergy, tolerance, sizeof(double),  __FILE__, __LINE__);
        assert_close(18, forcesS, (void*) parallelForces, tolerance, sizeof(double),  __FILE__, __LINE__);
        assert_close(6, virialS, (void*) parallelVirial, tolerance, sizeof(double),  __FILE__, __LINE__);
    }
    helpme_destroyD(pmeP); // This ensures that the PME object cleans up its MPI data BEFORE MPI_Finalize is called;

    MPI_Finalize();

    return 0;
}
