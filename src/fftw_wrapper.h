// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_FFTW_WRAPPER_H_
#define _HELPME_FFTW_WRAPPER_H_

#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <fftw3.h>
#include "memory.h"

namespace helpme {

/*!
 * \brief The FFTWTypes class is a placeholder to lookup function names and types in FFTW parlance by template.
 */
template <typename Real>
struct FFTWTypes {
    struct EmptyPlan {
        int unused;
    };
    using Plan = int;
    using Complex = std::complex<int>;
    static Plan makePlan4(size_t, void *, void *, int) { return 0; };
    static Plan makePlan5(size_t, void *, void *, int, int) { return 0; };
    static void execPlan1(Plan){};
    static void execPlan3(Plan, void *, void *){};
    static constexpr bool isImplemented = false;
    static constexpr decltype(&makePlan4) MakeRealToComplexPlan = &makePlan4;
    static constexpr decltype(&makePlan4) MakeComplexToRealPlan = &makePlan4;
    static constexpr decltype(&makePlan5) MakeComplexToComplexPlan = &makePlan5;
    static constexpr decltype(&execPlan3) ExecuteRealToComplexPlan = &execPlan3;
    static constexpr decltype(&execPlan3) ExecuteComplexToRealPlan = &execPlan3;
    static constexpr decltype(&execPlan3) ExecuteComplexToComplexPlan = &execPlan3;
    static constexpr decltype(&execPlan1) DestroyPlan = &execPlan1;
    static constexpr decltype(&execPlan1) CleanupFFTW = &execPlan1;
};

#if HAVE_FFTWF == 1
template <>
struct FFTWTypes<float> {
    using Plan = fftwf_plan;
    using Complex = fftwf_complex;
    static constexpr bool isImplemented = true;
    static constexpr decltype(&fftwf_plan_dft_r2c_1d) MakeRealToComplexPlan = &fftwf_plan_dft_r2c_1d;
    static constexpr decltype(&fftwf_plan_dft_c2r_1d) MakeComplexToRealPlan = &fftwf_plan_dft_c2r_1d;
    static constexpr decltype(&fftwf_plan_dft_1d) MakeComplexToComplexPlan = &fftwf_plan_dft_1d;
    static constexpr decltype(&fftwf_execute_dft_r2c) ExecuteRealToComplexPlan = &fftwf_execute_dft_r2c;
    static constexpr decltype(&fftwf_execute_dft_c2r) ExecuteComplexToRealPlan = &fftwf_execute_dft_c2r;
    static constexpr decltype(&fftwf_execute_dft) ExecuteComplexToComplexPlan = &fftwf_execute_dft;
    static constexpr decltype(&fftwf_destroy_plan) DestroyPlan = &fftwf_destroy_plan;
    static constexpr decltype(&fftwf_cleanup) CleanupFFTW = &fftwf_cleanup;
};
#endif  // HAVE_FFTWF

#if HAVE_FFTWD == 1
template <>
struct FFTWTypes<double> {
    using Plan = fftw_plan;
    using Complex = fftw_complex;
    static constexpr bool isImplemented = true;
    static constexpr decltype(&fftw_plan_dft_r2c_1d) MakeRealToComplexPlan = &fftw_plan_dft_r2c_1d;
    static constexpr decltype(&fftw_plan_dft_c2r_1d) MakeComplexToRealPlan = &fftw_plan_dft_c2r_1d;
    static constexpr decltype(&fftw_plan_dft_1d) MakeComplexToComplexPlan = &fftw_plan_dft_1d;
    static constexpr decltype(&fftw_execute_dft_r2c) ExecuteRealToComplexPlan = &fftw_execute_dft_r2c;
    static constexpr decltype(&fftw_execute_dft_c2r) ExecuteComplexToRealPlan = &fftw_execute_dft_c2r;
    static constexpr decltype(&fftw_execute_dft) ExecuteComplexToComplexPlan = &fftw_execute_dft;
    static constexpr decltype(&fftw_destroy_plan) DestroyPlan = &fftw_destroy_plan;
    static constexpr decltype(&fftw_cleanup) CleanupFFTW = &fftw_cleanup;
};
#endif  // HAVE_FFTWD

#if HAVE_FFTWL == 1
template <>
struct FFTWTypes<long double> {
    using Plan = fftwl_plan;
    using Complex = fftwl_complex;
    static constexpr bool isImplemented = true;
    static constexpr decltype(&fftwl_plan_dft_r2c_1d) MakeRealToComplexPlan = &fftwl_plan_dft_r2c_1d;
    static constexpr decltype(&fftwl_plan_dft_c2r_1d) MakeComplexToRealPlan = &fftwl_plan_dft_c2r_1d;
    static constexpr decltype(&fftwl_plan_dft_1d) MakeComplexToComplexPlan = &fftwl_plan_dft_1d;
    static constexpr decltype(&fftwl_execute_dft_r2c) ExecuteRealToComplexPlan = &fftwl_execute_dft_r2c;
    static constexpr decltype(&fftwl_execute_dft_c2r) ExecuteComplexToRealPlan = &fftwl_execute_dft_c2r;
    static constexpr decltype(&fftwl_execute_dft) ExecuteComplexToComplexPlan = &fftwl_execute_dft;
    static constexpr decltype(&fftwl_destroy_plan) DestroyPlan = &fftwl_destroy_plan;
    static constexpr decltype(&fftwl_cleanup) CleanupFFTW = &fftwl_cleanup;
};
#endif  // HAVE_FFTWL

/*!
 * \brief The FFTWWrapper class is a convenient wrapper to abstract away the details of different
 *        precision modes for FFTW, where the types and function names differ.
 */
template <typename Real>
class FFTWWrapper {
    using typeinfo = FFTWTypes<Real>;
    using Plan = typename typeinfo::Plan;
    using Complex = typename typeinfo::Complex;

   protected:
    /// An FFTW plan object, describing out of place complex to complex forward transforms.
    typename typeinfo::Plan forwardPlan_;
    /// An FFTW plan object, describing out of place complex to complex inverse transforms.
    typename typeinfo::Plan inversePlan_;
    /// An FFTW plan object, describing in place complex to complex forward transforms.
    typename typeinfo::Plan forwardInPlacePlan_;
    /// An FFTW plan object, describing in place complex to complex inverse transforms.
    typename typeinfo::Plan inverseInPlacePlan_;
    /// An FFTW plan object, describing out of place real to complex forward transforms.
    typename typeinfo::Plan realToComplexPlan_;
    /// An FFTW plan object, describing out of place complex to real inverse transforms.
    typename typeinfo::Plan complexToRealPlan_;
    /// The size of the real data.
    size_t fftDimension_;
    /// The flags to be passed to the FFTW plan creator, to determine startup cost.
    unsigned transformFlags_;

   public:
    FFTWWrapper() {}
    FFTWWrapper(size_t fftDimension) : fftDimension_(fftDimension), transformFlags_(FFTW_ESTIMATE) {
        if (!typeinfo::isImplemented) {
            throw std::runtime_error(
                "Attempting to call FFTW using a precision mode that has not been linked. "
                "Make sure that -DHAVE_FFTWF=1, -DHAVE_FFTWD=1 or -DHAVE_FFTWL=1 is added to the compiler flags"
                "for single, double and long double precision support, respectively.");
        }
        helpme::vector<Real> realTemp(fftDimension_);
        helpme::vector<std::complex<Real>> complexTemp1(fftDimension_);
        helpme::vector<std::complex<Real>> complexTemp2(fftDimension_);
        Real *realPtr = realTemp.data();
        Complex *complexPtr1 = reinterpret_cast<Complex *>(complexTemp1.data());
        Complex *complexPtr2 = reinterpret_cast<Complex *>(complexTemp2.data());
        forwardPlan_ =
            typeinfo::MakeComplexToComplexPlan(fftDimension_, complexPtr1, complexPtr2, FFTW_FORWARD, transformFlags_);
        inversePlan_ =
            typeinfo::MakeComplexToComplexPlan(fftDimension_, complexPtr1, complexPtr2, FFTW_BACKWARD, transformFlags_);
        forwardInPlacePlan_ =
            typeinfo::MakeComplexToComplexPlan(fftDimension_, complexPtr1, complexPtr1, FFTW_FORWARD, transformFlags_);
        inverseInPlacePlan_ =
            typeinfo::MakeComplexToComplexPlan(fftDimension_, complexPtr1, complexPtr1, FFTW_BACKWARD, transformFlags_);
        realToComplexPlan_ = typeinfo::MakeRealToComplexPlan(fftDimension_, realPtr, complexPtr1, transformFlags_);
        complexToRealPlan_ = typeinfo::MakeComplexToRealPlan(fftDimension_, complexPtr1, realPtr, transformFlags_);
    }

    /*!
     * \brief transform call FFTW to do an out of place complex to real FFT.
     * \param inBuffer the location of the input data.
     * \param outBuffer the location of the output data.
     */
    void transform(std::complex<Real> *inBuffer, Real *outBuffer) {
        typeinfo::ExecuteComplexToRealPlan(complexToRealPlan_, reinterpret_cast<Complex *>(inBuffer), outBuffer);
    }

    /*!
     * \brief transform call FFTW to do an out of place real to complex FFT.
     * \param inBuffer the location of the input data.
     * \param outBuffer the location of the output data.
     */
    void transform(Real *inBuffer, std::complex<Real> *outBuffer) {
        typeinfo::ExecuteRealToComplexPlan(realToComplexPlan_, inBuffer, reinterpret_cast<Complex *>(outBuffer));
    }

    /*!
     * \brief transform call FFTW to do an in place complex to complex FFT.
     * \param inPlaceBuffer the location of the input and output data.
     * \param direction either FFTW_FORWARD or FFTW_BACKWARD.
     */
    void transform(std::complex<Real> *inPlaceBuffer, int direction) {
        Complex *inPlacePtr = reinterpret_cast<Complex *>(inPlaceBuffer);
        switch (direction) {
            case FFTW_FORWARD:
                typeinfo::ExecuteComplexToComplexPlan(forwardInPlacePlan_, inPlacePtr, inPlacePtr);
                break;
            case FFTW_BACKWARD:
                typeinfo::ExecuteComplexToComplexPlan(inverseInPlacePlan_, inPlacePtr, inPlacePtr);
                break;
            default:
                throw std::runtime_error("Invalid FFTW transform passed to in place transform().");
        }
    }

    /*!
     * \brief transform call FFTW to do an out of place complex to complex FFT.
     * \param inBuffer the location of the input data.
     * \param outBuffer the location of the output data.
     * \param direction either FFTW_FORWARD or FFTW_BACKWARD.
     */
    void transform(std::complex<Real> *inBuffer, std::complex<Real> *outBuffer, int direction) {
        Complex *inPtr = reinterpret_cast<Complex *>(inBuffer);
        Complex *outPtr = reinterpret_cast<Complex *>(outBuffer);
        switch (direction) {
            case FFTW_FORWARD:
                typeinfo::ExecuteComplexToComplexPlan(forwardPlan_, inPtr, outPtr);
                break;
            case FFTW_BACKWARD:
                typeinfo::ExecuteComplexToComplexPlan(inversePlan_, inPtr, outPtr);
                break;
            default:
                throw std::runtime_error("Invalid FFTW transform passed to transform().");
        }
    }
};

}  // Namespace helpme
#endif  // Header guard
