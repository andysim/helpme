// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "helpme.h"
#include <chrono>

int main(int argc, char *argv[]) {
    bool useFloat = false;
    int rPower = 1;
    int nCalcs = 500;

    float scaleFactor = rPower == 1 ? 332.0716f : -1.0f;
    helpme::Matrix<double> coordsD("dhfr_coords.txt");
    helpme::Matrix<double> paramsD(rPower == 1 ? "dhfr_charges.txt" : "dhfr_c6s.txt");

    float kappa = 0.3;
    int gridX = 64;
    int gridY = 64;
    int gridZ = 64;
    int splineOrder = 4;

    auto startTime = std::chrono::system_clock::now();
    if (useFloat) {
        auto pme = std::unique_ptr<PMEInstanceF>(new PMEInstanceF());
        pme->setup(rPower, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 0);
        pme->setLatticeVectors(62.23f, 62.23f, 62.23f, 90.0f, 90.0f, 90.0f, PMEInstanceF::LatticeType::XAligned);
        helpme::Matrix<float> coordsF = coordsD.cast<float>();
        helpme::Matrix<float> paramsF = paramsD.cast<float>();
        helpme::Matrix<float> forces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<float> virial(6, 1);
        for (int n = 0; n < nCalcs; ++n) pme->computeEFVRec(0, paramsF, coordsF, forces, virial);
    } else {
        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
        pme->setup(rPower, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 0);
        pme->setLatticeVectors(62.23, 62.23, 62.23, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        helpme::Matrix<double> forces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<double> virial(6, 1);
        for (int n = 0; n < nCalcs; ++n) pme->computeEFVRec(0, paramsD, coordsD, forces, virial);
    }
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> runTime = endTime - startTime;
    std::cout << "Total run time: " << runTime.count() << std::endl;
}
