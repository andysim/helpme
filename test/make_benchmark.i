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

int main(int argc, char *argv[]) {
    // Defaults!
    bool useFloat = false;
    bool computeVirial = false;
    int rPower = 1;
    int nCalcs = 500;
    int gridA = DEFAULT_GRID_A;
    int gridB = DEFAULT_GRID_B;
    int gridC = DEFAULT_GRID_C;
    int maxKA = gridA;
    int maxKB = gridB;
    int maxKC = gridC;
    int splineOrder = 5;
    float beta = 0.3f;

    MPI_Init(NULL, NULL);
    int foundNumNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &foundNumNodes);
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // Parse
    static struct option long_options[] = {{"float", no_argument, 0, 'f'},
                                           {"virial", no_argument, 0, 'v'},
                                           {"grid", required_argument, 0, 'g'},
                                           {"ksum", required_argument, 0, 'k'},
                                           {"nruns", required_argument, 0, 'n'},
                                           {"rpower", required_argument, 0, 'r'},
                                           {"splineorder", required_argument, 0, 's'},
                                           {0, 0, 0, 0}};

    while (1) {
        /* getopt_long stores the option index here. */
        int option_index = 0;

        int c = getopt_long_only(argc, argv, "fg:k:n:r:s:v", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        /* Loop over user arguments */
        switch (c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0) break;
                printf("option %s", long_options[option_index].name);
                if (optarg) printf(" with arg %s", optarg);
                printf("\n");
                break;
            case 'b':
                beta = ::strtof(optarg, NULL);
                break;
            case 'f':
                useFloat = true;
                break;
            case 'g':
                gridA = gridB = gridC = ::strtol(optarg, NULL, 10);
                break;
            case 'k':
                maxKA = maxKB = maxKC = ::strtol(optarg, NULL, 10);
                break;
            case 'n':
                nCalcs = ::strtol(optarg, NULL, 10);
                break;
            case 'r':
                rPower = ::strtol(optarg, NULL, 10);
                break;
            case 's':
                splineOrder = ::strtol(optarg, NULL, 10);
                break;
            case 'v':
                computeVirial = true;
                break;
            default:
                abort();
        }
    }

    bool doCompressed = 2 * maxKA + 1 < gridA;

    int numNodesX, numNodesY, numNodesZ;
    // Print any remaining command line arguments (not options).
    if (argc - optind == 0) {
        numNodesX = numNodesY = numNodesZ = 1;
    } else if (argc - optind == 3) {
        numNodesX = ::strtol(argv[optind++], NULL, 10);
        numNodesY = ::strtol(argv[optind++], NULL, 10);
        numNodesZ = ::strtol(argv[optind++], NULL, 10);
    } else {
        throw std::runtime_error("There must be exactly 0 or 3 unlabeled options, corresonding to X,Y,Z node count.");
    }

    int numNodes = numNodesX * numNodesY * numNodesZ;
    if (foundNumNodes != numNodes) throw std::runtime_error("Mismatching number of nodes and X Y Z node dimensions!");

    int myNodeRankX = myRank % numNodesX;
    int myNodeRankY = (myRank % (numNodesY * numNodesX)) / numNodesX;
    int myNodeRankZ = myRank / (numNodesY * numNodesX);

    float scaleFactor = rPower == 1 ? 332.0716f : -1.0f;

    helpme::Matrix<double> coordsD(FILENAME "_coords.txt");
    helpme::Matrix<double> paramsD(rPower == 1 ? FILENAME "_charges.txt" : FILENAME "_c6s.txt");

    int nAtoms = coordsD.nRows();

    double boxDimX = BOX_SIZE_A;
    double boxDimY = BOX_SIZE_B;
    double boxDimZ = BOX_SIZE_C;

    double halfBoxDimX = boxDimX / 2;
    double halfBoxDimY = boxDimY / 2;
    double halfBoxDimZ = boxDimZ / 2;

    double myBoxDimX = boxDimX / numNodesX;
    double myBoxDimY = boxDimY / numNodesY;
    double myBoxDimZ = boxDimZ / numNodesZ;

    if (doCompressed) {
        // Filter the atoms so that each belongs to exactly one node, just to check it works.
        // If atoms are defined on multiple nodes, they're filtered out correctly.
        helpme::vector<double> myCoordsVec;
        helpme::vector<double> myChargeVec;
        for (int atom = 0; atom < coordsD.nRows(); ++atom) {
            const double *coords = coordsD[atom];
            double xCoord = coords[0] + halfBoxDimX;
            double yCoord = coords[1] + halfBoxDimY;
            double zCoord = coords[2] + halfBoxDimZ;
            if (xCoord > myNodeRankX * myBoxDimX && xCoord <= (myNodeRankX + 1) * myBoxDimX &&
                yCoord > myNodeRankY * myBoxDimY && yCoord <= (myNodeRankY + 1) * myBoxDimY &&
                zCoord > myNodeRankZ * myBoxDimZ && zCoord <= (myNodeRankZ + 1) * myBoxDimZ) {
                myCoordsVec.push_back(xCoord);
                myCoordsVec.push_back(yCoord);
                myCoordsVec.push_back(zCoord);
                myChargeVec.push_back(paramsD(atom, 0));
            }
        }
        coordsD = helpme::Matrix<double>(myCoordsVec.size() / 3, 3);
        paramsD = helpme::Matrix<double>(myChargeVec.size(), 1);
        std::copy(myCoordsVec.begin(), myCoordsVec.end(), coordsD[0]);
        std::copy(myChargeVec.begin(), myChargeVec.end(), paramsD[0]);
    }

    if (myRank == 0) {
        std::cout << "Algorithm: " << (doCompressed ? "Compressed" : "Conventional") << std::endl;
        std::cout << "Box Size: " << boxDimX << "x" << boxDimY << "x" << boxDimZ << std::endl;
        std::cout << "Number of Atoms " << nAtoms << std::endl;
        std::cout << "PME Parameter: " << beta << std::endl;
        std::cout << "R exponent: " << rPower << std::endl;
        std::cout << "Number of runs: " << nCalcs << std::endl;
        std::cout << "Node counts: " << numNodesX << "x" << numNodesY << "x" << numNodesZ << std::endl;
        std::cout << "Grid Dimensions: " << gridA << "x" << gridB << "x" << gridC << std::endl;
        std::cout << "Spline Order: " << splineOrder << std::endl;
        if(doCompressed) std::cout << "K sum Dimensions: " << maxKA << "x" << maxKB << "x" << maxKC << std::endl;
        std::cout << "Precision model: " << (useFloat ? "Float" : "Double") << std::endl;
    }

    auto startTime = std::chrono::system_clock::now();

    if (useFloat) {
        auto pme = std::unique_ptr<PMEInstanceF>(new PMEInstanceF());
        pme->setupCompressedParallel(rPower, beta, splineOrder, gridA, gridB, gridC, maxKA, maxKB, maxKC, scaleFactor,
                                     0, MPI_COMM_WORLD, PMEInstanceF::NodeOrder::ZYX, numNodesX, numNodesY, numNodesZ);
        pme->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceF::LatticeType::XAligned);
        helpme::Matrix<float> coordsF = coordsD.cast<float>();
        helpme::Matrix<float> paramsF = paramsD.cast<float>();
        helpme::Matrix<float> forces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<float> nodeForces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<float> virial(6, 1);
        helpme::Matrix<float> nodeVirial(6, 1);
        float nodeEnergy, energy;
        if (computeVirial) {
            for (int n = 0; n < nCalcs; ++n)
                nodeEnergy = pme->computeEFVRec(0, paramsF, coordsF, nodeForces, nodeVirial);
        } else {
            for (int n = 0; n < nCalcs; ++n) nodeEnergy = pme->computeEFRec(0, paramsF, coordsF, nodeForces);
        }
        MPI_Reduce(&nodeEnergy, &energy, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeForces[0], forces[0], 6 * 3, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeVirial[0], virial[0], 6, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        virial.applyOperationToEachElement([&](float &v) { v *= (1.0f / nCalcs); });
        if (myRank == 0) {
            std::cout << "Energy: " << std::setw(16) << std::setprecision(12) << energy << std::endl;
            if (computeVirial) std::cout << "Virial:" << std::endl << virial << std::endl;
        }
        pme.reset();
    } else {
        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
        pme->setupCompressedParallel(rPower, beta, splineOrder, gridA, gridB, gridC, maxKA, maxKB, maxKC, scaleFactor,
                                     0, MPI_COMM_WORLD, PMEInstanceD::NodeOrder::ZYX, numNodesX, numNodesY, numNodesZ);
        pme->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        helpme::Matrix<double> forces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<double> nodeForces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<double> virial(6, 1);
        helpme::Matrix<double> nodeVirial(6, 1);
        double nodeEnergy, energy;
        if (computeVirial) {
            for (int n = 0; n < nCalcs; ++n)
                nodeEnergy = pme->computeEFVRec(0, paramsD, coordsD, nodeForces, nodeVirial);
        } else {
            for (int n = 0; n < nCalcs; ++n) nodeEnergy = pme->computeEFRec(0, paramsD, coordsD, nodeForces);
        }
        MPI_Reduce(&nodeEnergy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeForces[0], forces[0], 6 * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeVirial[0], virial[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        virial.applyOperationToEachElement([&](double &v) { v *= (1.0 / nCalcs); });
        if (myRank == 0) {
            std::cout << "Energy: " << std::setw(16) << std::setprecision(12) << energy << std::endl;
            if (computeVirial) std::cout << "Virial:" << std::endl << virial << std::endl;
        }
        pme.reset();
    }
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> runTime = endTime - startTime;
    auto totalTime = runTime.count();
    if (myRank == 0)
        std::cout << "Total run time: " << totalTime << "s (" << 1000 * totalTime / nCalcs << " ms per calculation)"
                  << std::endl;

    MPI_Finalize();
}
