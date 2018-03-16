// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#if BUILD_STANDALONE
#include "helpme_standalone.h"
#else
#include "helpme.h"
#endif

#include <memory>

int main(int argc, char *argv[]) {
    // N.B. using std::make_unique is a cleaner approach for instantiation in C++14 and later
    // Instantiate double precision PME object
    helpme::Matrix<double> coordsD(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<double> chargesD({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    double scaleFactorD = 332.0716;
    helpme::Matrix<double> forcesD(6, 3);
    forcesD.setZero();

    auto pmeD = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
    pmeD->setup(1, 0.3, 6, 64, 64, 64, scaleFactorD, 1);
    pmeD->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
    double energyD = pmeD->computeEFRec(0, chargesD, coordsD, forcesD);

    // Instantiate single precision PME object
    helpme::Matrix<float> coordsF(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<float> chargesF({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    float scaleFactorF = 332.0716f;
    helpme::Matrix<float> forcesF(6, 3);
    forcesF.setZero();

    auto pmeF = std::unique_ptr<PMEInstanceF>(new PMEInstanceF);
    pmeF->setup(1, 0.3, 6, 64, 64, 64, scaleFactorF, 1);
    pmeF->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceF::LatticeType::XAligned);
    float energyF = pmeF->computeEFRec(0, chargesF, coordsF, forcesF);

    return 0;
}
