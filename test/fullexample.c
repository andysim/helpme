// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>

#include "helpme.h"
#include "print_results.h"

int main(int argc, char *argv[]) {
    int numThreads = argc > 1 ? atoi(argv[1]) : 1;
    printf("Num Threads: %d\n", numThreads);

    int atom;

    /*
     * Set up reference data for testing
     */
    double toleranceD = 1e-8;
    double toleranceF = 1e-4;
    double expectedEnergy = 5.864957414;
    double expectedForces[18] = {-1.20630693, -1.49522843, 12.65589187, 1.00695882,  0.88956328,  -5.08428301,
                                 0.69297661,  1.09547848,  -5.22771480, -2.28988057, -2.10832506, 10.18914165,
                                 0.81915340,  0.92013663,  -6.43738026, 0.97696467,  0.69833887,  -6.09492437};
    double expectedVirial[6] = {0.65613058, 0.49091167, 0.61109732, 2.26906257, 2.31925449, -10.04901641};
    double expectedPotential[24] = {1.18119329,  -0.72320559, -0.89641992, 7.58746515,  7.69247982,  -1.20738468,
                                    -1.06662264, 6.09626260,  8.73449635,  -0.83090721, -1.31352336, 6.26824317,
                                    -9.98483179, -1.37283008, -1.26398385, 6.10859811,  -3.50591589, -0.98219832,
                                    -1.10328133, 7.71868137,  -2.39904512, -1.17142047, -0.83733677, 7.30806279};

    /*
     * Instantiate double precision PME object
     */
    double coordsD[18] = {2.0, 2.0, 2.0, 2.5, 2.0, 3.0, 1.5, 2.0, 3.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, -0.5, 0.0, 1.0};
    double chargesD[6] = {-0.834, 0.417, 0.417, -0.834, 0.417, 0.417};
    double scaleFactorD = 332.0716;

    double energyD = 0;
    double forcesD[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double virialD[6] = {0, 0, 0, 0, 0, 0};
    double potentialAndGradientD[24] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    PMEInstance *pmeD = helpme_createD();
    helpme_setupD(pmeD, 1, 0.3, 5, 32, 32, 32, scaleFactorD, numThreads);
    helpme_set_lattice_vectorsD(pmeD, 20, 20, 20, 90, 90, 90, XAligned);
    // Compute just the energy
    print_resultsD(6, "Before E_recD", energyD, forcesD, virialD);
    energyD = helpme_compute_E_recD(pmeD, 6, 0, &chargesD[0], &coordsD[0]);
    print_resultsD(6, "After E_recD", energyD, forcesD, virialD);
    // Compute the energy and forces
    energyD = helpme_compute_EF_recD(pmeD, 6, 0, &chargesD[0], &coordsD[0], &forcesD[0]);
    print_resultsD(6, "After EF_recD", energyD, forcesD, virialD);
    // Compute the energy, forces, and virial
    energyD = helpme_compute_EFV_recD(pmeD, 6, 0, &chargesD[0], &coordsD[0], &forcesD[0], &virialD[0]);
    print_resultsD(6, "After EFV_recD", energyD, forcesD, virialD);
    // Compute the reciprocal space potential and its gradient at the atoms' coordinates
    helpme_compute_P_recD(pmeD, 6, 0, &chargesD[0], &coordsD[0], 6, &coordsD[0], 1, &potentialAndGradientD[0]);
    printf("Potential and its gradient:\n");
    for (atom = 0; atom < 6; ++atom)
        printf("%16.10f %16.10f %16.10f %16.10f\n", potentialAndGradientD[4 * atom + 0],
               potentialAndGradientD[4 * atom + 1], potentialAndGradientD[4 * atom + 2],
               potentialAndGradientD[4 * atom + 3]);
    printf("\n");

    assert_close(1, &expectedEnergy, (void *)&energyD, toleranceD, sizeof(double), __FILE__, __LINE__);
    assert_close(18, expectedForces, (void *)forcesD, toleranceD, sizeof(double), __FILE__, __LINE__);
    assert_close(6, expectedVirial, (void *)virialD, toleranceD, sizeof(double), __FILE__, __LINE__);
    assert_close(24, expectedPotential, (void *)potentialAndGradientD, toleranceD, sizeof(double), __FILE__, __LINE__);

    // Repeat the calculation using the compressed PME approximation
    energyD = 0;
    memset(forcesD, 0, 18 * sizeof(double));
    memset(virialD, 0, 6 * sizeof(double));
    memset(potentialAndGradientD, 0, 24 * sizeof(double));
    printf("\nCOMPRESSED\n\n");
    helpme_setup_compressedD(pmeD, 1, 0.3, 5, 32, 32, 32, 9, 9, 9, scaleFactorD, numThreads);
    helpme_set_lattice_vectorsD(pmeD, 20, 20, 20, 90, 90, 90, XAligned);
    // Compute just the energy
    print_resultsD(6, "Before E_recD", energyD, forcesD, virialD);
    energyD = helpme_compute_E_recD(pmeD, 6, 0, &chargesD[0], &coordsD[0]);
    print_resultsD(6, "After E_recD", energyD, forcesD, virialD);
    // Compute the energy and forces
    energyD = helpme_compute_EF_recD(pmeD, 6, 0, &chargesD[0], &coordsD[0], &forcesD[0]);
    print_resultsD(6, "After EF_recD", energyD, forcesD, virialD);
    // Compute the energy, forces, and virial
    energyD = helpme_compute_EFV_recD(pmeD, 6, 0, &chargesD[0], &coordsD[0], &forcesD[0], &virialD[0]);
    print_resultsD(6, "After EFV_recD", energyD, forcesD, virialD);
    // Compute the reciprocal space potential and its gradient at the atoms' coordinates
    helpme_compute_P_recD(pmeD, 6, 0, &chargesD[0], &coordsD[0], 6, &coordsD[0], 1, &potentialAndGradientD[0]);
    printf("Potential and its gradient:\n");
    for (atom = 0; atom < 6; ++atom)
        printf("%16.10f %16.10f %16.10f %16.10f\n", potentialAndGradientD[4 * atom + 0],
               potentialAndGradientD[4 * atom + 1], potentialAndGradientD[4 * atom + 2],
               potentialAndGradientD[4 * atom + 3]);
    printf("\n");

    assert_close(1, &expectedEnergy, (void *)&energyD, toleranceD, sizeof(double), __FILE__, __LINE__);
    assert_close(18, expectedForces, (void *)forcesD, toleranceD, sizeof(double), __FILE__, __LINE__);
    assert_close(6, expectedVirial, (void *)virialD, toleranceD, sizeof(double), __FILE__, __LINE__);
    assert_close(24, expectedPotential, (void *)potentialAndGradientD, toleranceD, sizeof(double), __FILE__, __LINE__);
    helpme_destroyD(pmeD);

    /*
     * Instantiate single precision PME object
     */
    float coordsF[18] = {2.0, 2.0, 2.0, 2.5, 2.0, 3.0, 1.5, 2.0, 3.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, -0.5, 0.0, 1.0};
    float chargesF[6] = {-0.834, 0.417, 0.417, -0.834, 0.417, 0.417};
    float scaleFactorF = 332.0716f;

    float energyF = 0;
    float forcesF[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float virialF[6] = {0, 0, 0, 0, 0, 0};
    float potentialAndGradientF[24] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    PMEInstance *pmeF = helpme_createF();
    helpme_setupF(pmeF, 1, 0.3, 5, 32, 32, 32, scaleFactorF, numThreads);
    helpme_set_lattice_vectorsF(pmeF, 20, 20, 20, 90, 90, 90, XAligned);
    // Compute just the energy
    print_resultsF(6, "Before E_recF", energyF, forcesF, virialF);
    energyF = helpme_compute_E_recF(pmeF, 6, 0, &chargesF[0], &coordsF[0]);
    print_resultsF(6, "After E_recF", energyF, forcesF, virialF);
    // Compute the energy and forces
    energyF = helpme_compute_EF_recF(pmeF, 6, 0, &chargesF[0], &coordsF[0], &forcesF[0]);
    print_resultsF(6, "After EF_recF", energyF, forcesF, virialF);
    // Compute the energy, forces, and virial
    energyF = helpme_compute_EFV_recF(pmeF, 6, 0, &chargesF[0], &coordsF[0], &forcesF[0], &virialF[0]);
    print_resultsF(6, "After EFV_recF", energyF, forcesF, virialF);
    // Compute the reciprocal space potential and its gradient at the atoms' coordinates
    helpme_compute_P_recF(pmeF, 6, 0, &chargesF[0], &coordsF[0], 6, &coordsF[0], 1, &potentialAndGradientF[0]);
    printf("Potential and its gradient:\n");
    for (atom = 0; atom < 6; ++atom)
        printf("%16.10f %16.10f %16.10f %16.10f\n", potentialAndGradientF[4 * atom + 0],
               potentialAndGradientF[4 * atom + 1], potentialAndGradientF[4 * atom + 2],
               potentialAndGradientF[4 * atom + 3]);
    printf("\n");

    assert_close(1, &expectedEnergy, (void *)&energyF, toleranceF, sizeof(float), __FILE__, __LINE__);
    assert_close(18, expectedForces, (void *)forcesF, toleranceF, sizeof(float), __FILE__, __LINE__);
    assert_close(6, expectedVirial, (void *)virialF, toleranceF, sizeof(float), __FILE__, __LINE__);
    assert_close(24, expectedPotential, (void *)potentialAndGradientF, toleranceF, sizeof(float), __FILE__, __LINE__);

    // Repeat the calculation using the compressed PME approximation
    energyF = 0;
    memset(forcesF, 0, 18 * sizeof(float));
    memset(virialF, 0, 6 * sizeof(float));
    memset(potentialAndGradientF, 0, 24 * sizeof(float));
    printf("\nCOMPRESSED\n\n");
    printf("TPE %f\n", scaleFactorF);
    helpme_setup_compressedF(pmeF, 1, 0.3, 5, 32, 32, 32, 9, 9, 9, scaleFactorF, numThreads);
    helpme_set_lattice_vectorsF(pmeF, 20, 20, 20, 90, 90, 90, XAligned);
    // Compute just the energy
    print_resultsF(6, "Before E_recF", energyF, forcesF, virialF);
    energyF = helpme_compute_E_recF(pmeF, 6, 0, &chargesF[0], &coordsF[0]);
    print_resultsF(6, "After E_recF", energyF, forcesF, virialF);
    // Compute the energy and forces
    energyF = helpme_compute_EF_recF(pmeF, 6, 0, &chargesF[0], &coordsF[0], &forcesF[0]);
    print_resultsF(6, "After EF_recF", energyF, forcesF, virialF);
    // Compute the energy, forces, and virial
    energyF = helpme_compute_EFV_recF(pmeF, 6, 0, &chargesF[0], &coordsF[0], &forcesF[0], &virialF[0]);
    print_resultsF(6, "After EFV_recF", energyF, forcesF, virialF);
    // Compute the reciprocal space potential and its gradient at the atoms' coordinates
    helpme_compute_P_recF(pmeF, 6, 0, &chargesF[0], &coordsF[0], 6, &coordsF[0], 1, &potentialAndGradientF[0]);
    printf("Potential and its gradient:\n");
    for (atom = 0; atom < 6; ++atom)
        printf("%16.10f %16.10f %16.10f %16.10f\n", potentialAndGradientF[4 * atom + 0],
               potentialAndGradientF[4 * atom + 1], potentialAndGradientF[4 * atom + 2],
               potentialAndGradientF[4 * atom + 3]);
    printf("\n");

    assert_close(1, &expectedEnergy, (void *)&energyF, toleranceF, sizeof(float), __FILE__, __LINE__);
    assert_close(18, expectedForces, (void *)forcesF, toleranceF, sizeof(float), __FILE__, __LINE__);
    assert_close(6, expectedVirial, (void *)virialF, toleranceF, sizeof(float), __FILE__, __LINE__);
    assert_close(24, expectedPotential, (void *)potentialAndGradientF, toleranceF, sizeof(float), __FILE__, __LINE__);
    helpme_destroyF(pmeF);
    return 0;
}
