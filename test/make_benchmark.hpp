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
    int numNodesX = 1;
    int numNodesY = 1;
    int numNodesZ = 1;

    MPI_Init(NULL, NULL);
    int foundNumNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &foundNumNodes);
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // Parse
    static struct option long_options[] = {
        {"beta", required_argument, 0, 'b'},   {"float", no_argument, 0, 'f'},
        {"grid", required_argument, 0, 'g'},   {"ksum", required_argument, 0, 'k'},
        {"nruns", required_argument, 0, 'n'},  {"parallel", required_argument, 0, 'p'},
        {"rpower", required_argument, 0, 'r'}, {"splineorder", required_argument, 0, 's'},
        {"virial", no_argument, 0, 'v'},       {0, 0, 0, 0}};

    while (1) {
        /* getopt_long stores the option index here. */
        int option_index = 0;

        int c = getopt_long_only(argc, argv, "b:fg:k:n:p:r:s:v", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        /* Loop over user arguments */
        switch (c) {
            case 'b':
                beta = ::strtof(optarg, NULL);
                break;
            case 'f':
                useFloat = true;
                break;
            case 'g':
                gridA = gridB = gridC = ::strtol(optarg, NULL, 10);
                if (optind < argc && argv[optind][0] != '-') {
                    // Look for a GridB spec
                    gridB = gridC = ::strtol(argv[optind++], NULL, 10);
                    if (optind < argc && argv[optind][0] != '-') {
                        // Look for a gridC spec
                        gridC = ::strtol(argv[optind++], NULL, 10);
                    }
                }
                break;
            case 'k':
                maxKA = maxKB = maxKC = ::strtol(optarg, NULL, 10);
                if (optind < argc && argv[optind][0] != '-') {
                    // Look for a KB spec
                    maxKB = maxKC = ::strtol(argv[optind++], NULL, 10);
                    if (optind < argc && argv[optind][0] != '-') {
                        // Look for a KC spec
                        maxKC = ::strtol(argv[optind++], NULL, 10);
                    }
                }
                break;
            case 'p':
                numNodesX = ::strtol(optarg, NULL, 10);
                if (optind < argc && argv[optind][0] != '-') {
                    // Look for a numNodesB spec
                    numNodesY = ::strtol(argv[optind++], NULL, 10);
                    if (optind < argc && argv[optind][0] != '-') {
                        // Look for a numNodesC spec
                        numNodesZ = ::strtol(argv[optind++], NULL, 10);
                    }
                }
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

    bool doCompressed = (2 * maxKA + 1 < gridA) && (2 * maxKB + 1 < gridB) && (2 * maxKC + 1 < gridC);

    // Print any remaining command line arguments (not options).
    if (argc - optind) {
        throw std::runtime_error("Unlabeled options found!");
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

    if (myRank == 0) {
        std::cout << "Algorithm: " << (doCompressed ? "Compressed" : "Conventional") << std::endl;
        std::cout << "Box Size: " << boxDimX << " x " << boxDimY << " x " << boxDimZ << std::endl;
        std::cout << "Number of Atoms " << nAtoms << std::endl;
        std::cout << "PME Parameter: " << beta << std::endl;
        std::cout << "R exponent: " << rPower << std::endl;
        std::cout << "Number of runs: " << nCalcs << std::endl;
        std::cout << "Node counts: " << numNodesX << " x " << numNodesY << " x " << numNodesZ << std::endl;
        std::cout << "Grid Dimensions: " << gridA << " x " << gridB << " x " << gridC << std::endl;
        std::cout << "Spline Order: " << splineOrder << std::endl;
        if (doCompressed) std::cout << "K sum Dimensions: " << maxKA << " x " << maxKB << " x " << maxKC << std::endl;
        std::cout << "Precision model: " << (useFloat ? "Float" : "Double") << std::endl;
    }

    auto startTime = std::chrono::system_clock::now();

    if (useFloat) {
        auto pme = std::unique_ptr<PMEInstanceF>(new PMEInstanceF());
        helpme::Matrix<float> coordsF = coordsD.cast<float>();
        helpme::Matrix<float> paramsF = paramsD.cast<float>();
        helpme::Matrix<float> forces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<float> nodeForces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<float> virial(6, 1);
        helpme::Matrix<float> nodeVirial(6, 1);
        float nodeEnergy, energy;
        if (computeVirial) {
            for (int n = 0; n < nCalcs; ++n) {
                pme->setupCompressedParallel(rPower, beta, splineOrder, gridA, gridB, gridC, maxKA, maxKB, maxKC,
                                             scaleFactor, 0, MPI_COMM_WORLD, PMEInstanceF::NodeOrder::ZYX, numNodesX,
                                             numNodesY, numNodesZ);
                pme->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceF::LatticeType::XAligned);
                nodeEnergy = pme->computeEFVRec(0, paramsF, coordsF, nodeForces, nodeVirial);
            }
        } else {
            for (int n = 0; n < nCalcs; ++n) {
                pme->setupCompressedParallel(rPower, beta, splineOrder, gridA, gridB, gridC, maxKA, maxKB, maxKC,
                                             scaleFactor, 0, MPI_COMM_WORLD, PMEInstanceF::NodeOrder::ZYX, numNodesX,
                                             numNodesY, numNodesZ);
                pme->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceF::LatticeType::XAligned);
                nodeEnergy = pme->computeEFRec(0, paramsF, coordsF, nodeForces);
            }
        }
        MPI_Reduce(&nodeEnergy, &energy, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeForces[0], forces[0], nAtoms * 3, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeVirial[0], virial[0], 6, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        virial.applyOperationToEachElement([&](float &v) { v *= (1.0f / nCalcs); });
        if (myRank == 0) {
            std::cout << "Energy: " << std::setw(16) << std::setprecision(12) << energy << std::endl;
            if (computeVirial) std::cout << "Virial:" << std::endl << virial << std::endl;
            forces.writeToFile("forces.dat");
        }
        pme.reset();
    } else {
        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
        helpme::Matrix<double> forces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<double> nodeForces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<double> virial(6, 1);
        helpme::Matrix<double> nodeVirial(6, 1);
        double nodeEnergy, energy;
        if (computeVirial) {
            for (int n = 0; n < nCalcs; ++n) {
                pme->setupCompressedParallel(rPower, beta, splineOrder, gridA, gridB, gridC, maxKA, maxKB, maxKC,
                                             scaleFactor, 0, MPI_COMM_WORLD, PMEInstanceD::NodeOrder::ZYX, numNodesX,
                                             numNodesY, numNodesZ);
                pme->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
                nodeEnergy = pme->computeEFVRec(0, paramsD, coordsD, nodeForces, nodeVirial);
            }
        } else {
            for (int n = 0; n < nCalcs; ++n) {
                pme->setupCompressedParallel(rPower, beta, splineOrder, gridA, gridB, gridC, maxKA, maxKB, maxKC,
                                             scaleFactor, 0, MPI_COMM_WORLD, PMEInstanceD::NodeOrder::ZYX, numNodesX,
                                             numNodesY, numNodesZ);
                pme->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
                nodeEnergy = pme->computeEFRec(0, paramsD, coordsD, nodeForces);
            }
        }
        MPI_Reduce(&nodeEnergy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeForces[0], forces[0], nAtoms * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nodeVirial[0], virial[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        virial.applyOperationToEachElement([&](double &v) { v *= (1.0 / nCalcs); });
        if (myRank == 0) {
            std::cout << "Energy: " << std::setw(16) << std::setprecision(12) << energy << std::endl;
            if (computeVirial) std::cout << "Virial:" << std::endl << virial << std::endl;
            forces.writeToFile("forces.dat");
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
