// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"
#include "fftw_wrapper.h"
#include "memory.h"

#include <algorithm>
#include <complex>
#include <iostream>
#include <iomanip>

template <typename Real>
bool isClose(const helpme::vector<Real> &expected, const helpme::vector<Real> &found, const double &tol) {
    return std::equal(expected.begin(), expected.end(), found.begin(),
                      [&tol](const Real &a, const Real &b) -> bool { return (std::abs(a - b) < tol); });
}

template <typename Real>
bool isClose(const helpme::vector<std::complex<Real>> &expected, const helpme::vector<std::complex<Real>> &found,
             const double &tol) {
    const Real *realFound = reinterpret_cast<const Real *>(found.data());
    const Real *realExpected = reinterpret_cast<const Real *>(expected.data());
    size_t size = 2 * (std::min(found.size(), expected.size()));
    return std::equal(realFound, realFound + size, realExpected,
                      [&tol](const Real &a, const Real &b) -> bool { return (std::abs(a - b) < tol); });
}

TEST_CASE("test the fftw wrapper class.") {
#if HAVE_FFTWF == 1
    SECTION("single precision tests") {
        const float TOL = 1e-6f;
        helpme::vector<float> input{0.1f, 0.3f, 0.6f, 0.9f};
        helpme::vector<float> refFromComplex{0.4f, 1.2f, 2.4f, 3.6f};
        helpme::vector<std::complex<float>> refComplex1 = {
            std::complex<float>(1.9f, 0.0f), std::complex<float>(-0.5f, 0.6f), std::complex<float>(-0.5f, 0.0f)};
        helpme::vector<std::complex<float>> complexIn = {
            std::complex<float>(1.0f, 0.0f), std::complex<float>(2.0f, -1.0f), std::complex<float>(0.0f, -1.0f),
            std::complex<float>(-1.0f, 2.0f)};
        helpme::vector<std::complex<float>> refComplexIn = {
            std::complex<float>(4.0f, 0.0f), std::complex<float>(8.0f, -4.0f), std::complex<float>(0.0f, -4.0f),
            std::complex<float>(-4.0f, 8.0f)};
        helpme::vector<std::complex<float>> refComplexToComplex = {
            std::complex<float>(2.0f, 0.0f), std::complex<float>(-2.0f, -2.0f), std::complex<float>(0.0f, -2.0f),
            std::complex<float>(4.0f, 4.0f)};
        helpme::vector<std::complex<float>> realToComplex(3);
        helpme::vector<float> realFromComplex(4);
        helpme::FFTWWrapper<float> fftHelper(4);
        fftHelper.transform(input.data(), realToComplex.data());
        REQUIRE(isClose<float>(refComplex1, realToComplex, TOL));
        fftHelper.transform(realToComplex.data(), realFromComplex.data());
        REQUIRE(isClose<float>(refFromComplex, realFromComplex, TOL));
        fftHelper.transform(complexIn.data(), FFTW_FORWARD);
        REQUIRE(isClose<float>(refComplexToComplex, complexIn, TOL));
        fftHelper.transform(complexIn.data(), FFTW_BACKWARD);
        REQUIRE(isClose<float>(refComplexIn, complexIn, TOL));
    }
#endif

#if HAVE_FFTWD == 1
    SECTION("double precision tests") {
        const double TOL = 1e-8;
        helpme::vector<double> input{0.1, 0.3, 0.6, 0.9};
        helpme::vector<double> refFromComplex{0.4, 1.2, 2.4, 3.6};
        helpme::vector<std::complex<double>> refComplex1 = {
            std::complex<double>(1.9, 0.0), std::complex<double>(-0.5, 0.6), std::complex<double>(-0.5, 0.0)};
        helpme::vector<std::complex<double>> complexIn = {
            std::complex<double>(1.0, 0.0), std::complex<double>(2.0, -1.0), std::complex<double>(0.0, -1.0),
            std::complex<double>(-1.0, 2.0)};
        helpme::vector<std::complex<double>> refComplexIn = {
            std::complex<double>(4.0, 0.0), std::complex<double>(8.0, -4.0), std::complex<double>(0.0, -4.0),
            std::complex<double>(-4.0, 8.0)};
        helpme::vector<std::complex<double>> refComplexToComplex = {
            std::complex<double>(2.0, 0.0), std::complex<double>(-2.0, -2.0), std::complex<double>(0.0, -2.0),
            std::complex<double>(4.0, 4.0)};
        helpme::vector<std::complex<double>> realToComplex(3);
        helpme::vector<double> realFromComplex(4);
        helpme::FFTWWrapper<double> fftHelper(4);
        fftHelper.transform(input.data(), realToComplex.data());
        REQUIRE(isClose<double>(refComplex1, realToComplex, TOL));
        fftHelper.transform(realToComplex.data(), realFromComplex.data());
        REQUIRE(isClose<double>(refFromComplex, realFromComplex, TOL));
        fftHelper.transform(complexIn.data(), FFTW_FORWARD);
        REQUIRE(isClose<double>(refComplexToComplex, complexIn, TOL));
        fftHelper.transform(complexIn.data(), FFTW_BACKWARD);
        REQUIRE(isClose<double>(refComplexIn, complexIn, TOL));
    }
#endif

#if HAVE_FFTWL == 1
    SECTION("long double precision tests") {
        const long double TOL = 1e-16l;
        helpme::vector<long double> input{0.1, 0.3, 0.6, 0.9};
        helpme::vector<long double> refFromComplex{0.4, 1.2, 2.4, 3.6};
        helpme::vector<std::complex<long double>> refComplex1 = {std::complex<long double>(1.9, 0.0),
                                                                 std::complex<long double>(-0.5, 0.6),
                                                                 std::complex<long double>(-0.5, 0.0)};
        helpme::vector<std::complex<long double>> complexIn = {
            std::complex<long double>(1.0, 0.0), std::complex<long double>(2.0, -1.0),
            std::complex<long double>(0.0, -1.0), std::complex<long double>(-1.0, 2.0)};
        helpme::vector<std::complex<long double>> refComplexIn = {
            std::complex<long double>(4.0, 0.0), std::complex<long double>(8.0, -4.0),
            std::complex<long double>(0.0, -4.0), std::complex<long double>(-4.0, 8.0)};
        helpme::vector<std::complex<long double>> refComplexToComplex = {
            std::complex<long double>(2.0, 0.0), std::complex<long double>(-2.0, -2.0),
            std::complex<long double>(0.0, -2.0), std::complex<long double>(4.0, 4.0)};
        helpme::vector<std::complex<long double>> realToComplex(3);
        helpme::vector<long double> realFromComplex(4);
        helpme::FFTWWrapper<long double> fftHelper(4);
        fftHelper.transform(input.data(), realToComplex.data());
        REQUIRE(isClose<long double>(refComplex1, realToComplex, TOL));
        fftHelper.transform(realToComplex.data(), realFromComplex.data());
        REQUIRE(isClose<long double>(refFromComplex, realFromComplex, TOL));
        fftHelper.transform(complexIn.data(), FFTW_FORWARD);
        REQUIRE(isClose<long double>(refComplexToComplex, complexIn, TOL));
        fftHelper.transform(complexIn.data(), FFTW_BACKWARD);
        REQUIRE(isClose<long double>(refComplexIn, complexIn, TOL));
    }
#endif

    SECTION("instantiate unsupported type") {
        REQUIRE_THROWS_WITH(helpme::FFTWWrapper<int>(4), Catch::Contains("precision mode"));
    }
}
