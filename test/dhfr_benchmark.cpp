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
    bool doCompressed = true;
    bool computeVirial = true;
    int rPower = 1;
    int nCalcs = 1;
    int numNodesX;
    int numNodesY;
    int numNodesZ;
    if (argc == 4) {
        numNodesX = ::atoi(argv[1]);
        numNodesY = ::atoi(argv[2]);
        numNodesZ = ::atoi(argv[3]);
    } else if (argc == 1) {
        numNodesX = 1;
        numNodesY = 1;
        numNodesZ = 1;
    } else {
        throw std::runtime_error(
            "This test should be run with exactly 3 arguments describing the number of X,Y and Z nodes.");
    }
    int numNodes = numNodesX * numNodesY * numNodesZ;

    MPI_Init(NULL, NULL);
    int foundNumNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &foundNumNodes);
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (foundNumNodes != numNodes) throw std::runtime_error("Mismatching number of nodes and X Y Z node dimensions!");

    int myNodeRankX = myRank % numNodesX;
    int myNodeRankY = (myRank % (numNodesY * numNodesX)) / numNodesX;
    int myNodeRankZ = myRank / (numNodesY * numNodesX);
    double boxDimX = 64;
    double boxDimY = 64;
    double boxDimZ = 64;
    double halfBoxDimX = boxDimX / 2;
    double halfBoxDimY = boxDimY / 2;
    double halfBoxDimZ = boxDimZ / 2;
    double myBoxDimX = boxDimX / numNodesX;
    double myBoxDimY = boxDimY / numNodesY;
    double myBoxDimZ = boxDimZ / numNodesZ;
    float scaleFactor = rPower == 1 ? 332.0716f : -1.0f;
    helpme::Matrix<double> coordsD("dhfr_coords.txt");
    helpme::Matrix<double> paramsD(rPower == 1 ? "dhfr_charges.txt" : "dhfr_c6s.txt");

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
    float kappa = 0.3;
    int gridX = 76;
    int gridY = 76;
    int gridZ = 76;
    int Ka = doCompressed ? 29 : gridX;
    int Kb = doCompressed ? 29 : gridY;
    int Kc = doCompressed ? 29 : gridZ;
    int splineOrder = 6;

    auto startTime = std::chrono::system_clock::now();

    if (useFloat) {
        auto pme = std::unique_ptr<PMEInstanceF>(new PMEInstanceF());
        // pme->setupCompressedParallel(rPower, kappa, splineOrder, gridX, gridY, gridZ, Ka, Kb, Kc, scaleFactor, 0,
        //                             MPI_COMM_WORLD, PMEInstanceF::NodeOrder::ZYX, numNodesX, numNodesY, numNodesZ);
        pme->setupParallel(rPower, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 0, MPI_COMM_WORLD,
                           PMEInstanceF::NodeOrder::ZYX, numNodesX, numNodesY, numNodesZ);
        pme->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceF::LatticeType::XAligned);
        helpme::Matrix<float> coordsF = coordsD.cast<float>();
        helpme::Matrix<float> paramsF = paramsD.cast<float>();
        helpme::Matrix<float> forces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<float> virial(6, 1);
        float energy;
        if (computeVirial) {
            for (int n = 0; n < nCalcs; ++n) energy = pme->computeEFVRec(0, paramsF, coordsF, forces, virial);
        } else {
            for (int n = 0; n < nCalcs; ++n) energy = pme->computeEFRec(0, paramsF, coordsF, forces);
        }
        std::cout << energy << std::endl;
        std::cout << virial << std::endl;
        pme.reset();
    } else {
        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
        pme->setupCompressedParallel(rPower, kappa, splineOrder, gridX, gridY, gridZ, Ka, Kb, Kc, scaleFactor, 0,
                                     MPI_COMM_WORLD, PMEInstanceD::NodeOrder::ZYX, numNodesX, numNodesY, numNodesZ);
        pme->setLatticeVectors(boxDimX, boxDimY, boxDimZ, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        helpme::Matrix<double> forces(coordsD.nRows(), coordsD.nCols());
        helpme::Matrix<double> virial(6, 1);
        double energy;
        if (computeVirial) {
            for (int n = 0; n < nCalcs; ++n) energy = pme->computeEFVRec(0, paramsD, coordsD, forces, virial);
        } else {
            for (int n = 0; n < nCalcs; ++n) energy = pme->computeEFRec(0, paramsD, coordsD, forces);
        }
        std::cout << energy << std::endl;
        std::cout << virial << std::endl;
        pme.reset();
    }
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> runTime = endTime - startTime;
    std::cout << "Total run time: " << runTime.count() << std::endl;

    MPI_Finalize();
}
