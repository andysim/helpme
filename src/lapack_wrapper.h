// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_LAPACK_WRAPPER_H_
#define _HELPME_LAPACK_WRAPPER_H_

#include <exception>
#include <complex>

#if FC_SYMBOL == 2
#define F_SGEEV sgeev_
#define F_SGESV sgesv_
#define F_DGEEV dgeev_
#define F_DGESV dgesv_
#define F_CGEEV cgeev_
#define F_CGESV cgesv_
#define F_ZGEEV zgeev_
#define F_ZGESV zgesv_
#elif FC_SYMBOL == 1
#define F_SGEEV sgeev
#define F_SGESV sgesv
#define F_DGEEV dgeev
#define F_DGESV dgesv
#define F_CGEEV cgeev
#define F_CGESV cgesv
#define F_ZGEEV zgeev
#define F_ZGESV zgesv
#elif FC_SYMBOL == 3
#define F_SGEEV SGEEV
#define F_SGESV SGESV
#define F_DGEEV DGEEV
#define F_DGESV DGESV
#define F_CGEEV CGEEV
#define F_CGESV CGESV
#define F_ZGEEV ZGEEV
#define F_ZGESV ZGESV
#elif FC_SYMBOL == 4
#define F_SGEEV SGEEV_
#define F_SGESV SGESV_
#define F_DGEEV DGEEV_
#define F_DGESV DGESV_
#define F_CGEEV CGEEV_
#define F_CGESV CGESV_
#define F_ZGEEV ZGEEV_
#define F_ZGESV ZGESV_
#endif

extern "C" {
extern void F_SGEEV(char *, char *, int *, float *, int *, float *, float *, float *, int *, float *, int *, float *,
                    int *, int *);
extern void F_DGEEV(char *, char *, int *, double *, int *, double *, double *, double *, int *, double *, int *,
                    double *, int *, int *);
extern void F_CGEEV(char *, char *, int *, std::complex<float> *, int *, std::complex<float> *, std::complex<float> *,
                    std::complex<float> *, int *, std::complex<float> *, int *, std::complex<float> *, int *, int *);
extern void F_ZGEEV(char *, char *, int *, std::complex<double> *, int *, std::complex<double> *,
                    std::complex<double> *, std::complex<double> *, int *, std::complex<double> *, int *,
                    std::complex<double> *, int *, int *);
}

namespace helpme {

static void C_SGEEV(char jobvl, char jobvr, int n, float *a, int lda, float *wr, float *wi, float *vl, int ldvl,
                    float *vr, int ldvr, float *work, int lwork, int *info) {
    ::F_SGEEV(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

static void C_DGEEV(char jobvl, char jobvr, int n, double *a, int lda, double *wr, double *wi, double *vl, int ldvl,
                    double *vr, int ldvr, double *work, int lwork, int *info) {
    ::F_DGEEV(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

static void C_CGEEV(char jobvl, char jobvr, int n, std::complex<float> *a, int lda, std::complex<float> *wr,
                    std::complex<float> *wi, std::complex<float> *vl, int ldvl, std::complex<float> *vr, int ldvr,
                    std::complex<float> *work, int lwork, int *info) {
    ::F_CGEEV(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

static void C_ZGEEV(char jobvl, char jobvr, int n, std::complex<double> *a, int lda, std::complex<double> *wr,
                    std::complex<double> *wi, std::complex<double> *vl, int ldvl, std::complex<double> *vr, int ldvr,
                    std::complex<double> *work, int lwork, int *info) {
    ::F_ZGEEV(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

template <typename Real>
using diagonalizerType =
    std::function<void(char, char, int, Real *, int, Real *, Real *, Real *, int, Real *, int, Real *, int, int *)>;

template <typename Real>
class LapackWrapper {
   public:
    static diagonalizerType<Real> diagonalizer() {
        throw std::runtime_error("Diagonalization is not implemented for the requested data type");
        return diagonalizerType<Real>();
    }
};

template <>
inline diagonalizerType<float> LapackWrapper<float>::diagonalizer() {
    return &C_SGEEV;
}
template <>
inline diagonalizerType<double> LapackWrapper<double>::diagonalizer() {
    return &C_DGEEV;
}
template <>
inline diagonalizerType<std::complex<float>> LapackWrapper<std::complex<float>>::diagonalizer() {
    return &C_CGEEV;
}
template <>
inline diagonalizerType<std::complex<double>> LapackWrapper<std::complex<double>>::diagonalizer() {
    return &C_ZGEEV;
}

}  // Namespace helpme
#endif  // Header guard
