// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_CARTESIANTRANSFORM_H_
#define _HELPME_CARTESIANTRANSFORM_H_

#include "matrix.h"

#include <vector>

namespace helpme {

static inline int cartesianAddress(int lx, int ly, int lz) {
    int l = lx + ly + lz;
    return lz * (2 * l - lz + 3) / 2 + ly;
}

/*!
 * \brief makeCartesianRotationMatrix builds a rotation matrix for unique Cartesian
 *        components with a given angular momentum.  The algorithm used here is the simple
 *        version (eq. 18) from D. M. Elking, J. Comp. Chem., 37 2067 (2016).  It's definitely
 *        not the fastest way to do it, but will be revisited if profiling shows it to be an issue.
 * \param angularMomentum the angular momentum of the rotation matrix desired.
 * \param transformer the matrix R to do the transform defined for a dipole as µ_new = R . µ_old.
 * \return the rotation matrix
 */
template <typename Real>
Matrix<Real> makeCartesianRotationMatrix(int angularMomentum, const Matrix<Real> &transformer) {
    Real R00 = transformer[0][0];
    Real R01 = transformer[0][1];
    Real R02 = transformer[0][2];
    Real R10 = transformer[1][0];
    Real R11 = transformer[1][1];
    Real R12 = transformer[1][2];
    Real R20 = transformer[2][0];
    Real R21 = transformer[2][1];
    Real R22 = transformer[2][2];
    int nComponents = (angularMomentum + 1) * (angularMomentum + 2) / 2;
    auto factorial = std::vector<int>(2 * angularMomentum + 1);
    factorial[0] = 1;
    for (int l = 1; l <= 2 * angularMomentum; ++l) factorial[l] = l * factorial[l - 1];
    Matrix<Real> R(nComponents, nComponents);
    for (int nz = 0; nz <= angularMomentum; ++nz) {
        for (int ny = 0; ny <= angularMomentum - nz; ++ny) {
            int nx = angularMomentum - ny - nz;
            for (int pz = 0; pz <= nx; ++pz) {
                for (int py = 0; py <= nx - pz; ++py) {
                    int px = nx - py - pz;
                    for (int qz = 0; qz <= ny; ++qz) {
                        for (int qy = 0; qy <= ny - qz; ++qy) {
                            int qx = ny - qy - qz;
                            for (int rz = 0; rz <= nz; ++rz) {
                                for (int ry = 0; ry <= nz - rz; ++ry) {
                                    int rx = nz - ry - rz;
                                    int mx = px + qx + rx;
                                    int my = py + qy + ry;
                                    int mz = pz + qz + rz;
                                    int m = mx + my + mz;
                                    if (m == angularMomentum) {
                                        Real normx = factorial[mx] / (factorial[px] * factorial[qx] * factorial[rx]);
                                        Real normy = factorial[my] / (factorial[py] * factorial[qy] * factorial[ry]);
                                        Real normz = factorial[mz] / (factorial[pz] * factorial[qz] * factorial[rz]);
                                        Real Rx = std::pow(R00, px) * std::pow(R10, py) * std::pow(R20, pz);
                                        Real Ry = std::pow(R01, qx) * std::pow(R11, qy) * std::pow(R21, qz);
                                        Real Rz = std::pow(R02, rx) * std::pow(R12, ry) * std::pow(R22, rz);
                                        Real term = normx * normy * normz * Rx * Ry * Rz;
                                        R[cartesianAddress(mx, my, mz)][cartesianAddress(nx, ny, nz)] += term;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return R;
}

/*!
 * \brief matrixVectorProduct A naive implementation of matrix-vector products, avoiding BLAS requirements (for now).
 * \param transformer the transformation matrix.
 * \param inputVector the vector to be transformed.
 * \param outputVector the transformed vector.
 */
template <typename Real>
void matrixVectorProduct(const Matrix<Real> &transformer, const Real *inputVector, Real *outputVector) {
    int dimension = transformer.nRows();
    for (int row = 0; row < dimension; ++row) {
        outputVector[row] = std::inner_product(inputVector, inputVector + dimension, transformer[row], Real(0));
    }
}

/*!
 * \brief cartesianTransform transforms a list of a cartesian quantities to a different basis.
 *        Assumes a list of quantities are to be transformed (in place) and all angular momentum
 *        components up to and including the specified maximum are present in ascending A.M. order.
 * \param maxAngularMomentum the angular momentum of the incoming quantity.
 * \param transformOnlyThisShell if true, only the shell with angular momentum specified will be transformed
 * \param transformer the matrix R to do the transform defined for a dipole as µ_new = R . µ_old.
 * \param transformee the quantity to be transformed, stored as nAtoms X nComponents, with
 *        components being the fast running index.
 */
template <typename Real>
Matrix<Real> cartesianTransform(int maxAngularMomentum, bool transformOnlyThisShell, const Matrix<Real> &transformer,
                                const Matrix<Real> &transformee) {
    Matrix<Real> transformed = transformee.clone();
    int offset = transformOnlyThisShell ? 0 : 1;
    int nAtoms = transformee.nRows();
    int firstShell = transformOnlyThisShell ? maxAngularMomentum : 1;
    for (int angularMomentum = firstShell; angularMomentum <= maxAngularMomentum; ++angularMomentum) {
        auto rotationMatrix = makeCartesianRotationMatrix(angularMomentum, transformer);
        for (int atom = 0; atom < nAtoms; ++atom) {
            const Real *inputData = transformee[atom];
            Real *outputData = transformed[atom];
            matrixVectorProduct(rotationMatrix, inputData + offset, outputData + offset);
        }
        offset += (angularMomentum + 1) * (angularMomentum + 2) / 2;
    }
    return transformed;
}

}  // Namespace helpme
#endif  // Header guard
