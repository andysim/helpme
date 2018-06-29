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

#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

template <typename T>
void printResults(std::string label, T e, const helpme::Matrix<T> &f, const helpme::Matrix<T> &v) {
    std::cout << label << std::endl;
    std::cout << "Energy = " << std::setw(16) << std::setprecision(10) << e << std::endl;
    std::cout << "Forces:" << std::endl << f << std::endl;
    std::cout << "Virial:" << std::endl << v << std::endl;
}

int main(int argc, char *argv[]) {
    /*
     * Some reference values for testing
     */
    const double toleranceD = 1e-8;
    const double toleranceF = 1e-4;
    double expectedEnergy = 5.864957414;
    helpme::Matrix<double> expectedForces({{-1.20630693, -1.49522843, 12.65589187},
                                           {1.00695882, 0.88956328, -5.08428301},
                                           {0.69297661, 1.09547848, -5.22771480},
                                           {-2.28988057, -2.10832506, 10.18914165},
                                           {0.81915340, 0.92013663, -6.43738026},
                                           {0.97696467, 0.69833887, -6.09492437}});
    helpme::Matrix<double> expectedVirial({{0.65613058, 0.49091167, 0.61109732, 2.26906257, 2.31925449, -10.04901641}});
    helpme::Matrix<double> expectedPotential({{1.18119329, -0.72320559, -0.89641992, 7.58746515},
                                              {7.69247982, -1.20738468, -1.06662264, 6.09626260},
                                              {8.73449635, -0.83090721, -1.31352336, 6.26824317},
                                              {-9.98483179, -1.37283008, -1.26398385, 6.10859811},
                                              {-3.50591589, -0.98219832, -1.10328133, 7.71868137},
                                              {-2.39904512, -1.17142047, -0.83733677, 7.30806279}});

    /*
     *  Instantiate double precision PME object
     */
    helpme::Matrix<double> coordsD(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<double> chargesD({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    double scaleFactorD = 332.0716;

    double energyD = 0;
    helpme::Matrix<double> forcesD(6, 3);
    helpme::Matrix<double> virialD(1, 6);
    helpme::Matrix<double> potentialAndGradientD(6, 4);

    auto pmeD = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
    pmeD->setup(1, 0.3, 5, 32, 32, 32, scaleFactorD, 1);
    pmeD->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
    // Compute just the energy
    printResults("Before computeEFRec double", energyD, forcesD, virialD);
    energyD = pmeD->computeERec(0, chargesD, coordsD);
    printResults("After computeEFRec double", energyD, forcesD, virialD);
    // Compute the energy and forces
    energyD = pmeD->computeEFRec(0, chargesD, coordsD, forcesD);
    printResults("After computeEFRec double", energyD, forcesD, virialD);
    // Compute the energy, forces, and virial
    energyD = pmeD->computeEFVRec(0, chargesD, coordsD, forcesD, virialD);
    printResults("After computeEFVRec double", energyD, forcesD, virialD);
    // Compute the reciprocal space potential and field at the atoms' coordinates
    pmeD->computePRec(0, chargesD, coordsD, coordsD, 1, potentialAndGradientD);
    std::cout << "Potential and its gradient:" << std::endl;
    std::cout << potentialAndGradientD << std::endl;

    assert((std::abs(energyD - expectedEnergy) < toleranceD));
    assert((potentialAndGradientD.almostEquals(expectedPotential, toleranceD)));
    assert((forcesD.almostEquals(expectedForces, toleranceD)));
    assert((virialD.almostEquals(expectedVirial, toleranceD)));

    /*
     *  Instantiate single precision PME object
     */
    helpme::Matrix<float> coordsF(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<float> chargesF({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    float scaleFactorF = 332.0716f;

    float energyF = 0;
    helpme::Matrix<float> forcesF(6, 3);
    helpme::Matrix<float> virialF(1, 6);
    helpme::Matrix<float> potentialAndGradientF(6, 4);

    auto pmeF = std::unique_ptr<PMEInstanceF>(new PMEInstanceF);
    pmeF->setup(1, 0.3, 5, 32, 32, 32, scaleFactorF, 1);
    pmeF->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceF::LatticeType::XAligned);
    // Compute just the energy
    printResults("Before computeEFRec float", energyF, forcesF, virialF);
    energyF = pmeF->computeERec(0, chargesF, coordsF);
    printResults("After computeEFRec float", energyF, forcesF, virialF);
    // Compute the energy and forces
    energyF = pmeF->computeEFRec(0, chargesF, coordsF, forcesF);
    printResults("After computeEFRec float", energyF, forcesF, virialF);
    // Compute the energy, forces, and virial
    energyF = pmeF->computeEFVRec(0, chargesF, coordsF, forcesF, virialF);
    printResults("After computeEFVRec float", energyF, forcesF, virialF);
    // Compute the reciprocal space potential and its gradient at the atoms' coordinates
    pmeF->computePRec(0, chargesF, coordsF, coordsF, 1, potentialAndGradientF);
    std::cout << "Potential and its gradient:" << std::endl;
    std::cout << potentialAndGradientF << std::endl;

    assert((std::abs(energyF - expectedEnergy) < toleranceF));
    assert((potentialAndGradientF.almostEquals(expectedPotential.cast<float>(), toleranceF)));
    assert((forcesF.almostEquals(expectedForces.cast<float>(), toleranceF)));
    assert((virialF.almostEquals(expectedVirial.cast<float>(), toleranceF)));
    return 0;
}
