// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "helpme.h"

int main(int argc, char *argv[]) {
    // Instantiate double precision PME object
    double coordsD[18] = {
          2.0, 2.0, 2.0,
          2.5, 2.0, 3.0,
          1.5, 2.0, 3.0,
          0.0, 0.0, 0.0,
          0.5, 0.0, 1.0,
         -0.5, 0.0, 1.0
    };
    double forcesD[18] = {
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0
    };
    double chargesD[6] = {-0.834, 0.417, 0.417, -0.834, 0.417, 0.417};
    double scaleFactorD = 332.0716;
    PMEInstance *pmeD = helpme_createD();
    helpme_setupD(pmeD, 1, 0.3, 5, 32, 32, 32, scaleFactorD, 1);
    helpme_set_lattice_vectorsD(pmeD, 20, 20, 20, 90, 90, 90, XAligned);
    double energyD = helpme_compute_EF_recD(pmeD, 6, 0, &chargesD[0], &coordsD[0], &forcesD[0]);

    // Instantiate single precision PME object
    float coordsF[18] = {
          2.0, 2.0, 2.0,
          2.5, 2.0, 3.0,
          1.5, 2.0, 3.0,
          0.0, 0.0, 0.0,
          0.5, 0.0, 1.0,
         -0.5, 0.0, 1.0
    };
    float forcesF[18] = {
          0.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f
    };
    float chargesF[6] = {-0.834, 0.417, 0.417, -0.834, 0.417, 0.417};
    float scaleFactorF = 332.0716f;
    PMEInstance *pmeF = helpme_createF();
    helpme_setupF(pmeF, 1, 0.3, 5, 32, 32, 32, scaleFactorF, 1);
    helpme_set_lattice_vectorsF(pmeF, 20, 20, 20, 90, 90, 90, XAligned);
    float energyF = helpme_compute_EF_recF(pmeF, 6, 0, &chargesF[0], &coordsF[0], &forcesF[0]);
    return 0;
}
