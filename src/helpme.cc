// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#include "helpme.h"

#include <iostream>

// C Wrappers

extern "C" {

typedef enum { XAligned = 0, ShapeMatrix = 1 } LatticeType;
typedef enum { ZYX = 0 } NodeOrder;

PMEInstanceD* helpme_createD() {
    try {
        return new PMEInstanceD();
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_createD" << std::endl;
        exit(1);
    }
}

PMEInstanceF* helpme_createF() {
    try {
        return new PMEInstanceF();
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_createF" << std::endl;
        exit(1);
    }
}

void helpme_destroyD(PMEInstanceD* pme) {
    try {
        delete pme;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_destroyD" << std::endl;
        exit(1);
    }
}

void helpme_destroyF(PMEInstanceF* pme) {
    try {
        delete pme;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_destroyF" << std::endl;
        exit(1);
    }
}

void helpme_setupD(PMEInstanceD* pme, short rPower, double kappa, int splineOrder, int aDim, int bDim, int cDim,
                   double scaleFactor, int nThreads) {
    try {
        pme->setup(rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setupD" << std::endl;
        exit(1);
    }
}

void helpme_setupF(PMEInstanceF* pme, short rPower, float kappa, int splineOrder, int aDim, int bDim, int cDim,
                   float scaleFactor, int nThreads) {
    try {
        pme->setup(rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setupF" << std::endl;
        exit(1);
    }
}

#if HAVE_MPI == 1
void helpme_setup_parallelD(PMEInstanceD* pme, int rPower, double kappa, int splineOrder, int dimA, int dimB, int dimC,
                            double scaleFactor, int nThreads, MPI_Comm communicator, NodeOrder nodeOrder, int numNodesA,
                            int numNodesB, int numNodesC) {
    try {
        pme->setupParallel(rPower, kappa, splineOrder, dimA, dimB, dimC, scaleFactor, nThreads, communicator,
                           PMEInstanceD::NodeOrder(nodeOrder), numNodesA, numNodesB, numNodesC);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setup_parallelD" << std::endl;
        exit(1);
    }
}

void helpme_setup_parallelF(PMEInstanceF* pme, int rPower, float kappa, int splineOrder, int dimA, int dimB, int dimC,
                            float scaleFactor, int nThreads, MPI_Comm communicator, NodeOrder nodeOrder, int numNodesA,
                            int numNodesB, int numNodesC) {
    try {
        pme->setupParallel(rPower, kappa, splineOrder, dimA, dimB, dimC, scaleFactor, nThreads, communicator,
                           PMEInstanceF::NodeOrder(nodeOrder), numNodesA, numNodesB, numNodesC);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setup_parallelF" << std::endl;
        exit(1);
    }
}

// Provide a wrapper to MPI_Comm_f2c; the C implementation may be a macro and is thus not callable from Fortran.
MPI_Comm f_MPI_Comm_f2c(int Fcomm) { return MPI_Comm_f2c(Fcomm); }

#endif

void helpme_set_lattice_vectorsD(PMEInstanceD* pme, double A, double B, double C, double alpha, double beta,
                                 double gamma, LatticeType latticeType) {
    try {
        pme->setLatticeVectors(A, B, C, alpha, beta, gamma, PMEInstanceD::LatticeType(latticeType));
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_set_lattice_vectorsD" << std::endl;
        exit(1);
    }
}

void helpme_set_lattice_vectorsF(PMEInstanceF* pme, float A, float B, float C, float alpha, float beta, float gamma,
                                 LatticeType latticeType) {
    try {
        pme->setLatticeVectors(A, B, C, alpha, beta, gamma, PMEInstanceF::LatticeType(latticeType));
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_set_lattice_vectorsF" << std::endl;
        exit(1);
    }
}

double helpme_compute_E_recD(PMEInstanceD* pme, int nAtoms, int parameterAngMom, double* parameters,
                             double* coordinates) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<double> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<double> coordMat(coordinates, nAtoms, 3);
        return pme->computeERec(parameterAngMom, paramMat, coordMat);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_E_recD" << std::endl;
        exit(1);
    }
}

float helpme_compute_E_recF(PMEInstanceF* pme, int nAtoms, int parameterAngMom, float* parameters, float* coordinates) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<float> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<float> coordMat(coordinates, nAtoms, 3);
        return pme->computeERec(parameterAngMom, paramMat, coordMat);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_E_recF" << std::endl;
        exit(1);
    }
}

double helpme_compute_EF_recD(PMEInstanceD* pme, int nAtoms, int parameterAngMom, double* parameters,
                              double* coordinates, double* forces) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<double> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<double> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<double> forceMat(forces, nAtoms, 3);
        return pme->computeEFRec(parameterAngMom, paramMat, coordMat, forceMat);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_EF_recD" << std::endl;
        exit(1);
    }
}

float helpme_compute_EF_recF(PMEInstanceF* pme, int nAtoms, int parameterAngMom, float* parameters, float* coordinates,
                             float* forces) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<float> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<float> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<float> forceMat(forces, nAtoms, 3);
        return pme->computeEFRec(parameterAngMom, paramMat, coordMat, forceMat);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_EF_recF" << std::endl;
        exit(1);
    }
}

double helpme_compute_EFV_recD(PMEInstanceD* pme, int nAtoms, int parameterAngMom, double* parameters,
                               double* coordinates, double* forces, double* virial) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<double> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<double> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<double> forceMat(forces, nAtoms, 3);
        helpme::Matrix<double> virialMat(virial, 1, 6);
        return pme->computeEFVRec(parameterAngMom, paramMat, coordMat, forceMat, virialMat);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_EFV_recD" << std::endl;
        exit(1);
    }
}

float helpme_compute_EFV_recF(PMEInstanceF* pme, int nAtoms, int parameterAngMom, float* parameters, float* coordinates,
                              float* forces, float* virial) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<float> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<float> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<float> forceMat(forces, nAtoms, 3);
        helpme::Matrix<float> virialMat(virial, 1, 6);
        return pme->computeEFVRec(parameterAngMom, paramMat, coordMat, forceMat, virialMat);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_EFV_recF" << std::endl;
        exit(1);
    }
}

void helpme_compute_P_recD(PMEInstanceD* pme, size_t nAtoms, int parameterAngMom, double* parameters,
                           double* coordinates, size_t nGridPoints, double* gridPoints, int derivativeLevel,
                           double* potential) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        int nDeriv = helpme::nCartesian(derivativeLevel);
        helpme::Matrix<double> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<double> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<double> gridMat(gridPoints, nGridPoints, 3);
        helpme::Matrix<double> potentialMat(potential, nGridPoints, nDeriv);
        pme->computePRec(parameterAngMom, paramMat, coordMat, gridMat, derivativeLevel, potentialMat);
        return;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_P_recD" << std::endl;
        exit(1);
    }
}

void helpme_compute_P_recF(PMEInstanceF* pme, size_t nAtoms, int parameterAngMom, float* parameters, float* coordinates,
                           size_t nGridPoints, float* gridPoints, int derivativeLevel, float* potential) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        int nDeriv = helpme::nCartesian(derivativeLevel);
        helpme::Matrix<float> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<float> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<float> gridMat(gridPoints, nGridPoints, 3);
        helpme::Matrix<float> potentialMat(potential, nGridPoints, nDeriv);
        pme->computePRec(parameterAngMom, paramMat, coordMat, gridMat, derivativeLevel, potentialMat);
        return;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_P_recF" << std::endl;
        exit(1);
    }
}
}
