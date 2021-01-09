// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#if defined D_MPI
#define HAVE_MPI 1
#endif
#if D_SCM_MATH_MKL == 1
#define HAVE_MKL 1
#define HAVE_FFTWD 1
// #define HAVE_FFTWF 1
#endif

#ifdef _WIN32
#define __restrict__ __restrict
#endif

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
        return nullptr;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_createD" << std::endl;
        return nullptr;
    }
}

PMEInstanceF* helpme_createF() {
    try {
        return new PMEInstanceF();
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_createF" << std::endl;
        return nullptr;
    }
}

int helpme_destroyD(PMEInstanceD* pme) {
    try {
        delete pme;
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_destroyD" << std::endl;
        return 1;
    }
}

int helpme_destroyF(PMEInstanceF* pme) {
    try {
        delete pme;
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_destroyF" << std::endl;
        return 1;
    }
}

int helpme_setupD(PMEInstanceD* pme, short rPower, double kappa, int splineOrder, int aDim, int bDim, int cDim,
                   double scaleFactor, int nThreads) {
    try {
        pme->setup(rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setupD" << std::endl;
        return 1;
    }
}

int helpme_setupF(PMEInstanceF* pme, short rPower, float kappa, int splineOrder, int aDim, int bDim, int cDim,
                   float scaleFactor, int nThreads) {
    try {
        pme->setup(rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setupF" << std::endl;
        return 1;
    }
}

int helpme_setup_compressedD(PMEInstanceD* pme, short rPower, double kappa, int splineOrder, int aDim, int bDim,
                              int cDim, int maxKA, int maxKB, int maxKC, double scaleFactor, int nThreads) {
    try {
        pme->setupCompressed(rPower, kappa, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC, scaleFactor, nThreads);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setup_compressedD" << std::endl;
        return 1;
    }
}

int helpme_setup_compressedF(PMEInstanceF* pme, short rPower, float kappa, int splineOrder, int aDim, int bDim,
                              int cDim, int maxKA, int maxKB, int maxKC, float scaleFactor, int nThreads) {
    try {
        pme->setupCompressed(rPower, kappa, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC, scaleFactor, nThreads);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setup_compressedF" << std::endl;
        return 1;
    }
}

#if HAVE_MPI == 1
int helpme_setup_parallelD(PMEInstanceD* pme, int rPower, double kappa, int splineOrder, int dimA, int dimB, int dimC,
                            double scaleFactor, int nThreads, MPI_Comm communicator, NodeOrder nodeOrder, int numNodesA,
                            int numNodesB, int numNodesC) {
    try {
        pme->setupParallel(rPower, kappa, splineOrder, dimA, dimB, dimC, scaleFactor, nThreads, communicator,
                           PMEInstanceD::NodeOrder(nodeOrder), numNodesA, numNodesB, numNodesC);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setup_parallelD" << std::endl;
        return 1;
    }
}

int helpme_setup_parallelF(PMEInstanceF* pme, int rPower, float kappa, int splineOrder, int dimA, int dimB, int dimC,
                            float scaleFactor, int nThreads, MPI_Comm communicator, NodeOrder nodeOrder, int numNodesA,
                            int numNodesB, int numNodesC) {
    try {
        pme->setupParallel(rPower, kappa, splineOrder, dimA, dimB, dimC, scaleFactor, nThreads, communicator,
                           PMEInstanceF::NodeOrder(nodeOrder), numNodesA, numNodesB, numNodesC);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setup_parallelF" << std::endl;
        return 1;
    }
}

int helpme_setup_compressed_parallelD(PMEInstanceD* pme, int rPower, double kappa, int splineOrder, int dimA, int dimB,
                                       int dimC, int maxKA, int maxKB, int maxKC, double scaleFactor, int nThreads,
                                       MPI_Comm communicator, NodeOrder nodeOrder, int numNodesA, int numNodesB,
                                       int numNodesC) {
    try {
        pme->setupCompressedParallel(rPower, kappa, splineOrder, dimA, dimB, dimC, maxKA, maxKB, maxKC, scaleFactor,
                                     nThreads, communicator, PMEInstanceD::NodeOrder(nodeOrder), numNodesA, numNodesB,
                                     numNodesC);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setup_parallelD" << std::endl;
        return 1;
    }
}

int helpme_setup_compressed_parallelF(PMEInstanceF* pme, int rPower, float kappa, int splineOrder, int dimA, int dimB,
                                       int dimC, int maxKA, int maxKB, int maxKC, float scaleFactor, int nThreads,
                                       MPI_Comm communicator, NodeOrder nodeOrder, int numNodesA, int numNodesB,
                                       int numNodesC) {
    try {
        pme->setupCompressedParallel(rPower, kappa, splineOrder, dimA, dimB, dimC, maxKA, maxKB, maxKC, scaleFactor,
                                     nThreads, communicator, PMEInstanceF::NodeOrder(nodeOrder), numNodesA, numNodesB,
                                     numNodesC);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_setup_parallelF" << std::endl;
        return 1;
    }
}

// Provide a wrapper to MPI_Comm_f2c; the C implementation may be a macro and is thus not callable from Fortran.
MPI_Comm f_MPI_Comm_f2c(int Fcomm) { return MPI_Comm_f2c(Fcomm); }

#endif

int helpme_set_latticeD(PMEInstanceD* pme, double *matrix) {
    try {
        helpme::Matrix<double> latticeMat(matrix, 3, 3);
        pme->setLattice(latticeMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_set_lattice_vectorsD" << std::endl;
        return 1;
    }
}

int helpme_set_latticeF(PMEInstanceF* pme, float *matrix) {
    try {
        helpme::Matrix<float> latticeMat(matrix, 3, 3);
        pme->setLattice(latticeMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_set_lattice_vectorsF" << std::endl;
        return 1;
    }
}

int helpme_set_lattice_vectorsD(PMEInstanceD* pme, double A, double B, double C, double alpha, double beta,
                                 double gamma, LatticeType latticeType) {
    try {
        pme->setLatticeVectors(A, B, C, alpha, beta, gamma, PMEInstanceD::LatticeType(latticeType));
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_set_lattice_vectorsD" << std::endl;
        return 1;
    }
}

int helpme_set_lattice_vectorsF(PMEInstanceF* pme, float A, float B, float C, float alpha, float beta, float gamma,
                                 LatticeType latticeType) {
    try {
        pme->setLatticeVectors(A, B, C, alpha, beta, gamma, PMEInstanceF::LatticeType(latticeType));
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_set_lattice_vectorsF" << std::endl;
        return 1;
    }
}

int helpme_compute_E_recD(PMEInstanceD* pme, int nAtoms, int parameterAngMom, double* parameters, double* coordinates, 
                          double* energy) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<double> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<double> coordMat(coordinates, nAtoms, 3);
        *energy = pme->computeERec(parameterAngMom, paramMat, coordMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_E_recD" << std::endl;
        return 1;
    }
}

int helpme_compute_E_recF(PMEInstanceF* pme, int nAtoms, int parameterAngMom, float* parameters, float* coordinates, 
                          float* energy) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<float> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<float> coordMat(coordinates, nAtoms, 3);
        *energy = pme->computeERec(parameterAngMom, paramMat, coordMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_E_recF" << std::endl;
        return 1;
    }
}

int helpme_compute_EF_recD(PMEInstanceD* pme, int nAtoms, int parameterAngMom, double* parameters, double* coordinates, 
                           double* energy, double* forces) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<double> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<double> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<double> forceMat(forces, nAtoms, 3);
        *energy = pme->computeEFRec(parameterAngMom, paramMat, coordMat, forceMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_EF_recD" << std::endl;
        return 1;
    }
}

int helpme_compute_EF_recF(PMEInstanceF* pme, int nAtoms, int parameterAngMom, float* parameters, float* coordinates,
                           float* energy, float* forces) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<float> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<float> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<float> forceMat(forces, nAtoms, 3);
        *energy = pme->computeEFRec(parameterAngMom, paramMat, coordMat, forceMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_EF_recF" << std::endl;
        return 1;
    }
}

int helpme_compute_EFV_recD(PMEInstanceD* pme, int nAtoms, int parameterAngMom, double* parameters, double* coordinates, 
                            double* energy, double* forces, double* virial) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<double> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<double> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<double> forceMat(forces, nAtoms, 3);
        helpme::Matrix<double> virialMat(virial, 1, 6);
        *energy = pme->computeEFVRec(parameterAngMom, paramMat, coordMat, forceMat, virialMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_EFV_recD" << std::endl;
        return 1;
    }
}

int helpme_compute_EFV_recF(PMEInstanceF* pme, int nAtoms, int parameterAngMom, float* parameters, float* coordinates,
                            float* energy, float* forces, float* virial) {
    try {
        int nParam = helpme::nCartesian(parameterAngMom);
        helpme::Matrix<float> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<float> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<float> forceMat(forces, nAtoms, 3);
        helpme::Matrix<float> virialMat(virial, 1, 6);
        *energy = pme->computeEFVRec(parameterAngMom, paramMat, coordMat, forceMat, virialMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_EFV_recF" << std::endl;
        return 1;
    }
}

int helpme_compute_P_recD(PMEInstanceD* pme, size_t nAtoms, int parameterAngMom, double* parameters, double* coordinates, 
                          size_t nGridPoints, double* gridPoints, int derivativeLevel, double* potential) {
    try {
        int nParam = 0;
        if (parameterAngMom >= 0)
            nParam = helpme::nCartesian(parameterAngMom);
        else
            nParam = helpme::nCartesian(std::abs(parameterAngMom)) - helpme::nCartesian(std::abs(parameterAngMom)-1);
        int nDeriv = 0;
        if (derivativeLevel >= 0)
            nDeriv = helpme::nCartesian(derivativeLevel);
        else
            nDeriv = helpme::nCartesian(std::abs(derivativeLevel)) - helpme::nCartesian(std::abs(derivativeLevel)-1);
        helpme::Matrix<double> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<double> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<double> gridMat(gridPoints, nGridPoints, 3);
        helpme::Matrix<double> potentialMat(potential, nGridPoints, nDeriv);
        pme->computePRec(parameterAngMom, paramMat, coordMat, gridMat, derivativeLevel, potentialMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_P_recD" << std::endl;
        return 1;
    }
}

int helpme_compute_P_recF(PMEInstanceF* pme, size_t nAtoms, int parameterAngMom, float* parameters, float* coordinates,
                          size_t nGridPoints, float* gridPoints, int derivativeLevel, float* potential) {
    try {
        int nParam = 0;
        if (parameterAngMom >= 0)
            nParam = helpme::nCartesian(parameterAngMom);
        else
            nParam = helpme::nCartesian(std::abs(parameterAngMom)) - helpme::nCartesian(std::abs(parameterAngMom)-1);
        int nDeriv = 0;
        if (derivativeLevel >= 0)
            nDeriv = helpme::nCartesian(derivativeLevel);
        else
            nDeriv = helpme::nCartesian(std::abs(derivativeLevel)) - helpme::nCartesian(std::abs(derivativeLevel)-1);
        helpme::Matrix<float> paramMat(parameters, nAtoms, nParam);
        helpme::Matrix<float> coordMat(coordinates, nAtoms, 3);
        helpme::Matrix<float> gridMat(gridPoints, nGridPoints, 3);
        helpme::Matrix<float> potentialMat(potential, nGridPoints, nDeriv);
        pme->computePRec(parameterAngMom, paramMat, coordMat, gridMat, derivativeLevel, potentialMat);
        return 0;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occured in helpme_compute_P_recF" << std::endl;
        return 1;
    }
}
}
