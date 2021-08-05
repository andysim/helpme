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
#include <stdlib.h>
#include <getopt.h>
#include <iostream>

int main(int argc, char *argv[]) {
    // Defaults!
    int rPower = 1;
    int gridA = 64;
    int gridB = 64;
    int gridC = 64;
    int maxKA = 72;
    int maxKB = 72;
    int maxKC = 72;
    int splineOrder = 5;
    double beta = 0.3;

    int numNodesX;
    int numNodesY;
    int numNodesZ;
    int numThreads;
    int maxNumAtomsToTest;
    if (argc == 5) {
        numNodesX = ::atoi(argv[1]);
        numNodesY = ::atoi(argv[2]);
        numNodesZ = ::atoi(argv[3]);
        numThreads = ::atoi(argv[4]);
        maxNumAtomsToTest = 0;
    } else if (argc == 6) {
        numNodesX = ::atoi(argv[1]);
        numNodesY = ::atoi(argv[2]);
        numNodesZ = ::atoi(argv[3]);
        numThreads = ::atoi(argv[4]);
        maxNumAtomsToTest = ::atoi(argv[5]);
    } else {
        std::cout << "You must provide numNodesX numNodesY numNodesZ numThreads [maxNumAtoms] as arguments!"
                  << std::endl;
        exit(1);
    }

    int foundNumNodes = 1;
    int myRank = 0;
#if HAVE_MPI == 1
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &foundNumNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif

    bool doCompressed = (2 * maxKA + 1 < gridA) && (2 * maxKB + 1 < gridB) && (2 * maxKC + 1 < gridC);

    int numNodes = numNodesX * numNodesY * numNodesZ;
    if (foundNumNodes != numNodes) throw std::runtime_error("Mismatching number of nodes and X Y Z node dimensions!");

    int myNodeRankX = myRank % numNodesX;
    int myNodeRankY = (myRank % (numNodesY * numNodesX)) / numNodesX;
    int myNodeRankZ = myRank / (numNodesY * numNodesX);

    double scaleFactor = rPower == 1 ? 332.0716 : -1.0;

    helpme::Matrix<double> coordsD("dhfr_coords.txt");
    helpme::Matrix<double> paramsD(rPower == 1 ? "dhfr_charges.txt" : "dhfr_c6s.txt");

    int nAtoms = coordsD.nRows();
    if (maxNumAtomsToTest == 0) maxNumAtomsToTest = nAtoms;

    double boxDimX = 64.0;
    double boxDimY = 64.0;
    double boxDimZ = 64.0;

    double halfBoxDimX = boxDimX / 2;
    double halfBoxDimY = boxDimY / 2;
    double halfBoxDimZ = boxDimZ / 2;

    double myBoxDimX = boxDimX / numNodesX;
    double myBoxDimY = boxDimY / numNodesY;
    double myBoxDimZ = boxDimZ / numNodesZ;

    if (myRank == 0) {
        std::cout << "Threads per node: " << numThreads << std::endl;
        std::cout << "Algorithm: " << (doCompressed ? "Compressed" : "Conventional") << std::endl;
        std::cout << "Box Size: " << boxDimX << " x " << boxDimY << " x " << boxDimZ << std::endl;
        std::cout << "Number of Atoms " << nAtoms << std::endl;
        std::cout << "PME Parameter: " << beta << std::endl;
        std::cout << "R exponent: " << rPower << std::endl;
        std::cout << "Node counts: " << numNodesX << " x " << numNodesY << " x " << numNodesZ << std::endl;
        std::cout << "Grid Dimensions: " << gridA << " x " << gridB << " x " << gridC << std::endl;
        std::cout << "Spline Order: " << splineOrder << std::endl;
        if (doCompressed) std::cout << "K sum Dimensions: " << maxKA << " x " << maxKB << " x " << maxKC << std::endl;
    }

    auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
    auto pmeSerial = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
    helpme::Matrix<double> forces(coordsD.nRows(), coordsD.nCols());
    helpme::Matrix<double> nodeForces(coordsD.nRows(), coordsD.nCols());
    helpme::Matrix<double> virial(6, 1);
    helpme::Matrix<double> nodeVirial(6, 1);
    double nodeEnergy, energy;
    if (numNodes == 1) {
        pme->setupCompressed(rPower, beta, splineOrder, gridA, gridB, gridC, maxKA, maxKB, maxKC, scaleFactor,
                             numThreads);
    } else {
#if HAVE_MPI == 1
        pme->setupCompressedParallel(rPower, beta, splineOrder, gridA, gridB, gridC, maxKA, maxKB, maxKC, scaleFactor,
                                     numThreads, MPI_COMM_WORLD, PMEInstanceD::NodeOrder::ZYX, numNodesX, numNodesY,
                                     numNodesZ);
#endif
    }
    pme->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
    nodeEnergy = pme->computeEFVRec(0, paramsD, coordsD, nodeForces, nodeVirial);

#if HAVE_MPI == 1
    MPI_Allreduce(&nodeEnergy, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(nodeForces[0], forces[0], nAtoms * 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(nodeVirial[0], virial[0], 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (myRank == 0) {
        std::cout << "Energy: " << std::setw(16) << std::setprecision(12) << energy << std::endl;
        std::cout << "Virial:" << std::endl << virial << std::endl;
    }

    pmeSerial->setupCompressed(rPower, beta, splineOrder, gridA, gridB, gridC, maxKA, maxKB, maxKC, scaleFactor, 0);
    pmeSerial->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
    int nTerms = std::min(maxNumAtomsToTest, nAtoms);
    std::vector<int> atomList(nTerms);
    if (myRank == 0) {
        for (int n = 0; n < nTerms; ++n) atomList[n] = n;
        std::random_shuffle(atomList.begin(), atomList.end());
    }
#if HAVE_MPI == 1
    MPI_Bcast(atomList.data(), nTerms, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    helpme::Matrix<double> findifForces(nTerms, 3);
    helpme::Matrix<double> forceErrors(nTerms, 3);
    double delta = 1e-5;
    double nodeMaxError = 0;
    for (int n = myRank; n < nTerms; n += numNodes) {
        int atom = atomList[n];
        for (int xyz = 0; xyz < 3; ++xyz) {
            coordsD[atom][xyz] += delta;
            double ep = pmeSerial->computeERec(0, paramsD, coordsD);
            coordsD[atom][xyz] -= 2 * delta;
            double em = pmeSerial->computeERec(0, paramsD, coordsD);
            coordsD[atom][xyz] += delta;
            findifForces[n][xyz] = (em - ep) / (2 * delta);
            nodeMaxError = std::max(nodeMaxError, std::abs(findifForces[n][xyz] - forces[atom][xyz]));
        }
    }
#if HAVE_MPI == 1
    double maxError = 0;
    MPI_Allreduce(findifForces[0], forceErrors[0], nTerms * 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&nodeMaxError, &maxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#else
    double maxError = nodeMaxError;
#endif
    if (myRank == 0) {
        for (int n = 0; n < nTerms; ++n) {
            int atom = atomList[n];
            forceErrors[n][0] -= forces[atom][0];
            forceErrors[n][1] -= forces[atom][1];
            forceErrors[n][2] -= forces[atom][2];
            std::cout << std::setw(8) << atom << helpme::stringify(forceErrors[n], 3, 3);
        }
        std::cout << "Maximum error: " << maxError << std::endl;
    }

    pme.reset();
#if HAVE_MPI == 1
    MPI_Finalize();
#endif
}
