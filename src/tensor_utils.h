// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_TENSOR_UTILS_H_
#define _HELPME_TENSOR_UTILS_H_

#if HAVE_BLAS == 1
extern "C" {
extern void dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *,
                   int *);
extern void sgemm_(char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *,
                   int *);
}
#endif

namespace helpme {

/*!
 * \brief Sorts a 3D tensor stored contiguously as ABC into CBA order.
 * \param abcPtr the address of the incoming ABC ordered tensor.
 * \param aDimension the dimension of the A index.
 * \param bDimension the dimension of the B index.
 * \param cDimension the dimension of the C index.
 * \param cbaPtr the address of the outgoing CBA ordered tensor.
 * \param nThreads the number of parallel threads to use.
 */
template <typename Real>
void permuteABCtoCBA(Real const *__restrict__ abcPtr, int const aDimension, int const bDimension, int const cDimension,
                     Real *__restrict__ cbaPtr, size_t nThreads = 1) {
#pragma omp parallel for num_threads(nThreads)
    for (int C = 0; C <= -1 + cDimension; ++C)
        for (int B = 0; B <= -1 + bDimension; ++B)
            for (int A = 0; A <= -1 + aDimension; ++A)
                cbaPtr[aDimension * bDimension * C + aDimension * B + A] =
                    abcPtr[cDimension * bDimension * A + cDimension * B + C];
}

/*!
 * \brief Sorts a 3D tensor stored contiguously as ABC into ACB order.
 * \param abcPtr the address of the incoming ABC ordered tensor.
 * \param aDimension the dimension of the A index.
 * \param bDimension the dimension of the B index.
 * \param cDimension the dimension of the C index.
 * \param acbPtr the address of the outgoing ACB ordered tensor.
 * \param nThreads the number of parallel threads to use.
 */
template <typename Real>
void permuteABCtoACB(Real const *__restrict__ abcPtr, int const aDimension, int const bDimension, int const cDimension,
                     Real *__restrict__ acbPtr, size_t nThreads = 1) {
#pragma omp parallel for num_threads(nThreads)
    for (int A = 0; A <= -1 + aDimension; ++A)
        for (int C = 0; C <= -1 + cDimension; ++C)
            for (int B = 0; B <= -1 + bDimension; ++B)
                acbPtr[bDimension * cDimension * A + bDimension * C + B] =
                    abcPtr[cDimension * bDimension * A + cDimension * B + C];
}

/*!
 * \brief Contracts an ABxC tensor with a DxC tensor, to produce an ABxD quantity.
 * \param abcPtr the address of the incoming ABxC tensor.
 * \param dcPtr the address of the incoming DxC tensor.
 * \param abDimension the dimension of the AB index.
 * \param cDimension the dimension of the C index.
 * \param dDimension the dimension of the D index.
 * \param abdPtr the address of the outgoing ABD tensor.
 */
template <typename Real>
void contractABxCWithDxC(Real const *__restrict__ abcPtr, Real const *__restrict__ dcPtr, int const abDimension,
                         int const cDimension, int const dDimension, Real *__restrict__ abdPtr) {
    Real acc_C;
    for (int AB = 0; AB <= -1 + abDimension; ++AB) {
        for (int D = 0; D <= -1 + dDimension; ++D) {
            acc_C = 0;
            for (int C = 0; C <= -1 + cDimension; ++C)
                acc_C = acc_C + abcPtr[cDimension * AB + C] * dcPtr[cDimension * D + C];
            abdPtr[dDimension * AB + D] = acc_C;
        }
    }
}

#if HAVE_BLAS == 1
template <>
void contractABxCWithDxC<float>(float const *__restrict__ abcPtr, float const *__restrict__ dcPtr,
                                int const abDimension, int const cDimension, int const dDimension,
                                float *__restrict__ abdPtr) {
    if (abDimension == 0 || cDimension == 0 || dDimension == 0) return;

    char transB = 't';
    char transA = 'n';
    float alpha = 1;
    float beta = 0;
    sgemm_(&transB, &transA, const_cast<int *>(&dDimension), const_cast<int *>(&abDimension),
           const_cast<int *>(&cDimension), &alpha, const_cast<float *>(dcPtr), const_cast<int *>(&cDimension),
           const_cast<float *>(abcPtr), const_cast<int *>(&cDimension), &beta, abdPtr, const_cast<int *>(&dDimension));
}

template <>
void contractABxCWithDxC<double>(double const *__restrict__ abcPtr, double const *__restrict__ dcPtr,
                                 int const abDimension, int const cDimension, int const dDimension,
                                 double *__restrict__ abdPtr) {
    if (abDimension == 0 || cDimension == 0 || dDimension == 0) return;

    char transB = 't';
    char transA = 'n';
    double alpha = 1;
    double beta = 0;
    dgemm_(&transB, &transA, const_cast<int *>(&dDimension), const_cast<int *>(&abDimension),
           const_cast<int *>(&cDimension), &alpha, const_cast<double *>(dcPtr), const_cast<int *>(&cDimension),
           const_cast<double *>(abcPtr), const_cast<int *>(&cDimension), &beta, abdPtr, const_cast<int *>(&dDimension));
}
#endif

}  // Namespace helpme
#endif  // Header guard
