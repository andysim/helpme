// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#include <stdio.h>

#include "helpme.h"

void print_resultsD(int natoms, char *label, double e, double f[], double v[]){
    printf("%s\n", label);
    printf("Energy = %16.10f\n", e);
    printf("Forces:\n");
    int atom;
    for(atom = 0; atom < natoms; ++atom)
       printf("%16.10f %16.10f %16.10f\n", f[3*atom], f[3*atom+1], f[3*atom+2]);
    printf("Virial:\n%16.10f %16.10f %16.10f %16.10f %16.10f %16.10f\n\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

void print_resultsF(int natoms, char *label, float e, float f[], float v[]){
    printf("%s\n", label);
    printf("Energy = %16.10f\n", e);
    printf("Forces:\n");
    int atom;
    for(atom = 0; atom < natoms; ++atom)
       printf("%16.10f %16.10f %16.10f\n", f[3*atom], f[3*atom+1], f[3*atom+2]);
    printf("Virial:\n%16.10f %16.10f %16.10f %16.10f %16.10f %16.10f\n\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}


int main(int argc, char *argv[]) {
    /*
     * Instantiate double precision PME object
     */
    double coordsD[18] = {
          2.0, 2.0, 2.0,
          2.5, 2.0, 3.0,
          1.5, 2.0, 3.0,
          0.0, 0.0, 0.0,
          0.5, 0.0, 1.0,
         -0.5, 0.0, 1.0
    };
    double chargesD[6] = {-0.834, 0.417, 0.417, -0.834, 0.417, 0.417};
    double scaleFactorD = 332.0716;

    double energyD = 0;
    double forcesD[18] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    double virialD[6] = {0, 0, 0, 0, 0, 0};

    PMEInstance *pmeD = helpme_createD();
    helpme_setupD(pmeD, 1, 0.3, 5, 32, 32, 32, scaleFactorD, 1);
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
    helpme_destroyD(pmeD);

    /*
     * Instantiate single precision PME object
     */
    float coordsF[18] = {
          2.0, 2.0, 2.0,
          2.5, 2.0, 3.0,
          1.5, 2.0, 3.0,
          0.0, 0.0, 0.0,
          0.5, 0.0, 1.0,
         -0.5, 0.0, 1.0
    };
    float chargesF[6] = {-0.834, 0.417, 0.417, -0.834, 0.417, 0.417};
    float scaleFactorF = 332.0716f;

    float energyF = 0;
    float forcesF[18] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float virialF[6] = {0, 0, 0, 0, 0, 0};

    PMEInstance *pmeF = helpme_createF();
    helpme_setupF(pmeF, 1, 0.3, 5, 32, 32, 32, scaleFactorF, 1);
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
    helpme_destroyF(pmeF);

    return 0;
}
