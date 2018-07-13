// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
//
// The code for Jacobi diagonalization is taken (with minimal modification) from
//
// http://www.mymathlib.com/c_source/matrices/eigen/jacobi_cyclic_method.c
//
#ifndef _HELPME_LAPACK_WRAPPER_H_
#define _HELPME_LAPACK_WRAPPER_H_

#include <cmath>
#include <limits>

namespace helpme {
////////////////////////////////////////////////////////////////////////////////
//  void Jacobi_Cyclic_Method                                                 //
//            (Real eigenvalues[], Real *eigenvectors, Real *A, int n)  //
//                                                                            //
//  Description:                                                              //
//     Find the eigenvalues and eigenvectors of a symmetric n x n matrix A    //
//     using the Jacobi method. Upon return, the input matrix A will have     //
//     been modified.                                                         //
//     The Jacobi procedure for finding the eigenvalues and eigenvectors of a //
//     symmetric matrix A is based on finding a similarity transformation     //
//     which diagonalizes A.  The similarity transformation is given by a     //
//     product of a sequence of orthogonal (rotation) matrices each of which  //
//     annihilates an off-diagonal element and its transpose.  The rotation   //
//     effects only the rows and columns containing the off-diagonal element  //
//     and its transpose, i.e. if a[i][j] is an off-diagonal element, then    //
//     the orthogonal transformation rotates rows a[i][] and a[j][], and      //
//     equivalently it rotates columns a[][i] and a[][j], so that a[i][j] = 0 //
//     and a[j][i] = 0.                                                       //
//     The cyclic Jacobi method considers the off-diagonal elements in the    //
//     following order: (0,1),(0,2),...,(0,n-1),(1,2),...,(n-2,n-1).  If the  //
//     the magnitude of the off-diagonal element is greater than a treshold,  //
//     then a rotation is performed to annihilate that off-diagnonal element. //
//     The process described above is called a sweep.  After a sweep has been //
//     completed, the threshold is lowered and another sweep is performed     //
//     with the new threshold. This process is completed until the final      //
//     sweep is performed with the final threshold.                           //
//     The orthogonal transformation which annihilates the matrix element     //
//     a[k][m], k != m, is Q = q[i][j], where q[i][j] = 0 if i != j, i,j != k //
//     i,j != m and q[i][j] = 1 if i = j, i,j != k, i,j != m, q[k][k] =       //
//     q[m][m] = cos(phi), q[k][m] = -sin(phi), and q[m][k] = sin(phi), where //
//     the angle phi is determined by requiring a[k][m] -> 0.  This condition //
//     on the angle phi is equivalent to                                      //
//               cot(2 phi) = 0.5 * (a[k][k] - a[m][m]) / a[k][m]             //
//     Since tan(2 phi) = 2 tan(phi) / (1 - tan(phi)^2),                      //
//               tan(phi)^2 + 2cot(2 phi) * tan(phi) - 1 = 0.                 //
//     Solving for tan(phi), choosing the solution with smallest magnitude,   //
//       tan(phi) = - cot(2 phi) + sgn(cot(2 phi)) sqrt(cot(2phi)^2 + 1).     //
//     Then cos(phi)^2 = 1 / (1 + tan(phi)^2) and sin(phi)^2 = 1 - cos(phi)^2 //
//     Finally by taking the sqrts and assigning the sign to the sin the same //
//     as that of the tan, the orthogonal transformation Q is determined.     //
//     Let A" be the matrix obtained from the matrix A by applying the        //
//     similarity transformation Q, since Q is orthogonal, A" = Q'AQ, where Q'//
//     is the transpose of Q (which is the same as the inverse of Q).  Then   //
//         a"[i][j] = Q'[i][p] a[p][q] Q[q][j] = Q[p][i] a[p][q] Q[q][j],     //
//     where repeated indices are summed over.                                //
//     If i is not equal to either k or m, then Q[i][j] is the Kronecker      //
//     delta.   So if both i and j are not equal to either k or m,            //
//                                a"[i][j] = a[i][j].                         //
//     If i = k, j = k,                                                       //
//        a"[k][k] =                                                          //
//           a[k][k]*cos(phi)^2 + a[k][m]*sin(2 phi) + a[m][m]*sin(phi)^2     //
//     If i = k, j = m,                                                       //
//        a"[k][m] = a"[m][k] = 0 =                                           //
//           a[k][m]*cos(2 phi) + 0.5 * (a[m][m] - a[k][k])*sin(2 phi)        //
//     If i = k, j != k or m,                                                 //
//        a"[k][j] = a"[j][k] = a[k][j] * cos(phi) + a[m][j] * sin(phi)       //
//     If i = m, j = k, a"[m][k] = 0                                          //
//     If i = m, j = m,                                                       //
//        a"[m][m] =                                                          //
//           a[m][m]*cos(phi)^2 - a[k][m]*sin(2 phi) + a[k][k]*sin(phi)^2     //
//     If i= m, j != k or m,                                                  //
//        a"[m][j] = a"[j][m] = a[m][j] * cos(phi) - a[k][j] * sin(phi)       //
//                                                                            //
//     If X is the matrix of normalized eigenvectors stored so that the ith   //
//     column corresponds to the ith eigenvalue, then AX = X Lamda, where     //
//     Lambda is the diagonal matrix with the ith eigenvalue stored at        //
//     Lambda[i][i], i.e. X'AX = Lambda and X is orthogonal, the eigenvectors //
//     are normalized and orthogonal.  So, X = Q1 Q2 ... Qs, where Qi is      //
//     the ith orthogonal matrix,  i.e. X can be recursively approximated by  //
//     the recursion relation X" = X Q, where Q is the orthogonal matrix and  //
//     the initial estimate for X is the identity matrix.                     //
//     If j = k, then x"[i][k] = x[i][k] * cos(phi) + x[i][m] * sin(phi),     //
//     if j = m, then x"[i][m] = x[i][m] * cos(phi) - x[i][k] * sin(phi), and //
//     if j != k and j != m, then x"[i][j] = x[i][j].                         //
//                                                                            //
//  Arguments:                                                                //
//     Real  eigenvalues                                                      //
//        Array of dimension n, which upon return contains the eigenvalues of //
//        the matrix A.                                                       //
//     Real* eigenvectors                                                     //
//        Matrix of eigenvectors, the ith column of which contains an         //
//        eigenvector corresponding to the ith eigenvalue in the array        //
//        eigenvalues.                                                        //
//     Real* A                                                                //
//        Pointer to the first element of the symmetric n x n matrix A. The   //
//        input matrix A is modified during the process.                      //
//     int     n                                                              //
//        The dimension of the array eigenvalues, number of columns and rows  //
//        of the matrices eigenvectors and A.                                 //
//                                                                            //
//  Return Values:                                                            //
//     Function is of type void.                                              //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     Real A[N][N], Real eigenvalues[N], Real eigenvectors[N][N]             //
//                                                                            //
//     (your code to initialize the matrix A )                                //
//                                                                            //
//     JacobiCyclicDiagonalization(eigenvalues, (Real*)eigenvectors,          //
//                                                          (Real *) A, N);   //
////////////////////////////////////////////////////////////////////////////////

template <typename Real>
void JacobiCyclicDiagonalization(Real *eigenvalues, Real *eigenvectors, const Real *A, int n) {
    int i, j, k, m;
    Real *pAk, *pAm, *p_r, *p_e;
    Real threshold_norm;
    Real threshold;
    Real tan_phi, sin_phi, cos_phi, tan2_phi, sin2_phi, cos2_phi;
    Real sin_2phi, cos_2phi, cot_2phi;
    Real dum1;
    Real dum2;
    Real dum3;
    Real max;

    // Take care of trivial cases

    if (n < 1) return;
    if (n == 1) {
        eigenvalues[0] = *A;
        *eigenvectors = 1;
        return;
    }

    // Initialize the eigenvalues to the identity matrix.

    for (p_e = eigenvectors, i = 0; i < n; i++)
        for (j = 0; j < n; p_e++, j++)
            if (i == j)
                *p_e = 1;
            else
                *p_e = 0;

    // Calculate the threshold and threshold_norm.

    for (threshold = 0, pAk = const_cast<Real *>(A), i = 0; i < (n - 1); pAk += n, i++)
        for (j = i + 1; j < n; j++) threshold += *(pAk + j) * *(pAk + j);
    threshold = sqrt(threshold + threshold);
    threshold_norm = threshold * std::numeric_limits<Real>::epsilon();
    max = threshold + 1;
    while (threshold > threshold_norm) {
        threshold /= 10;
        if (max < threshold) continue;
        max = 0;
        for (pAk = const_cast<Real *>(A), k = 0; k < (n - 1); pAk += n, k++) {
            for (pAm = pAk + n, m = k + 1; m < n; pAm += n, m++) {
                if (std::abs(*(pAk + m)) < threshold) continue;

                // Calculate the sin and cos of the rotation angle which
                // annihilates A[k][m].

                cot_2phi = 0.5f * (*(pAk + k) - *(pAm + m)) / *(pAk + m);
                dum1 = sqrt(cot_2phi * cot_2phi + 1);
                if (cot_2phi < 0) dum1 = -dum1;
                tan_phi = -cot_2phi + dum1;
                tan2_phi = tan_phi * tan_phi;
                sin2_phi = tan2_phi / (1 + tan2_phi);
                cos2_phi = 1 - sin2_phi;
                sin_phi = sqrt(sin2_phi);
                if (tan_phi < 0) sin_phi = -sin_phi;
                cos_phi = sqrt(cos2_phi);
                sin_2phi = 2 * sin_phi * cos_phi;
                cos_2phi = cos2_phi - sin2_phi;

                // Rotate columns k and m for both the matrix A
                //     and the matrix of eigenvectors.

                p_r = const_cast<Real *>(A);
                dum1 = *(pAk + k);
                dum2 = *(pAm + m);
                dum3 = *(pAk + m);
                *(pAk + k) = dum1 * cos2_phi + dum2 * sin2_phi + dum3 * sin_2phi;
                *(pAm + m) = dum1 * sin2_phi + dum2 * cos2_phi - dum3 * sin_2phi;
                *(pAk + m) = 0;
                *(pAm + k) = 0;
                for (i = 0; i < n; p_r += n, i++) {
                    if ((i == k) || (i == m)) continue;
                    if (i < k)
                        dum1 = *(p_r + k);
                    else
                        dum1 = *(pAk + i);
                    if (i < m)
                        dum2 = *(p_r + m);
                    else
                        dum2 = *(pAm + i);
                    dum3 = dum1 * cos_phi + dum2 * sin_phi;
                    if (i < k)
                        *(p_r + k) = dum3;
                    else
                        *(pAk + i) = dum3;
                    dum3 = -dum1 * sin_phi + dum2 * cos_phi;
                    if (i < m)
                        *(p_r + m) = dum3;
                    else
                        *(pAm + i) = dum3;
                }
                for (p_e = eigenvectors, i = 0; i < n; p_e += n, i++) {
                    dum1 = *(p_e + k);
                    dum2 = *(p_e + m);
                    *(p_e + k) = dum1 * cos_phi + dum2 * sin_phi;
                    *(p_e + m) = -dum1 * sin_phi + dum2 * cos_phi;
                }
            }
            for (i = 0; i < n; i++)
                if (i == k)
                    continue;
                else if (max < std::abs(*(pAk + i)))
                    max = std::abs(*(pAk + i));
        }
    }
    for (pAk = const_cast<Real *>(A), k = 0; k < n; pAk += n, k++) eigenvalues[k] = *(pAk + k);
}

}  // Namespace helpme
#endif  // Header guard
