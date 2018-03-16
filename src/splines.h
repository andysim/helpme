// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_SPLINES_H_
#define _HELPME_SPLINES_H_

#include "matrix.h"

/*!
 * \file splines.h
 * \brief Contains the C++ implementation of a cardinal B-Splines.
 */

namespace helpme {

/*!
 * \class BSpline
 * \brief A class to compute cardinal B-splines. This code can compute arbitrary-order B-splines of
 *        arbitrary derivative level, subject to the usual constraint that an order m spline is
 *        differentiable m-2 times.
 * \tparam Real the floating point type to use for arithmetic.
 */
template <typename Real>
class BSpline {
   protected:
    /// The order of this B-spline.
    short order_;
    /// The maximum derivative level for this B-spline.
    short derivativeLevel_;
    /// B-Splines with rows corresponding to derivative level, and columns to spline component.
    Matrix<Real> splines_;
    /// The grid point at which to start interpolation.
    short startingGridPoint_;

    /// Makes B-Spline array.
    void makeSplineInPlace(Real *array, Real val, short n) {
        Real denom = (Real)1 / (n - 1);
        array[n - 1] = denom * val * array[n - 2];
        for (short j = 1; j < n - 1; ++j)
            array[n - j - 1] = denom * ((val + j) * array[n - j - 2] + (n - j - val) * array[n - j - 1]);
        array[0] *= denom * (1 - val);
    }

    /// Takes BSpline derivative.
    void differentiateSpline(const Real *array, Real *dArray, short n) {
        dArray[0] = -array[0];
        for (short j = 1; j < n - 1; ++j) dArray[j] = array[j - 1] - array[j];
        dArray[n - 1] = array[n - 2];
    }

   public:
    /// The B-splines and their derivatives
    BSpline(short start, Real value, short order, short derivativeLevel)
        : order_(order),
          derivativeLevel_(derivativeLevel),
          splines_(derivativeLevel + 1, order),
          startingGridPoint_(start) {
        splines_.setZero();
        splines_(0, 0) = 1 - value;
        splines_(0, 1) = value;
        for (short m = 1; m < order - 1; ++m) {
            makeSplineInPlace(splines_[0], value, m + 2);
            if (m >= order - derivativeLevel_ - 2) {
                short currentDerivative = order_ - m - 2;
                for (short l = 0; l < currentDerivative; ++l)
                    differentiateSpline(splines_[l], splines_[l + 1], m + 2 + currentDerivative);
            }
        }
    }

    BSpline() {}

    /*!
     * \brief The modulus of the B-Spline in Fourier space.
     * \param gridDim the dimension of the grid in the dimension this spline is to be used.
     * \return a gridDim long vector containing the inverse of the Fourier space spline moduli.
     */
    helpme::vector<Real> invSplineModuli(short gridDim) {
        helpme::vector<Real> splineMods(gridDim, 0);
        Real prefac = 2.0 * M_PI / gridDim;
        for (int i = 0; i < gridDim; ++i) {
            Real real = 0.0;
            Real imag = 0.0;
            for (int j = 0; j < order_; ++j) {
                Real exparg = i * j * prefac;
                Real jSpline = splines_(0, j);
                real += jSpline * cos(exparg);
                imag += jSpline * sin(exparg);
            }
            splineMods[i] = real * real + imag * imag;
        }

        // Correct tiny values.
        constexpr Real EPS = 1e-7;
        if (splineMods[0] < EPS) splineMods[0] = 0.5 * splineMods[1];
        for (int i = 0; i < gridDim - 1; ++i)
            if (splineMods[i] < EPS) splineMods[i] = 0.5 * (splineMods[i - 1] + splineMods[i + 1]);
        if (splineMods[gridDim - 1] < EPS) splineMods[gridDim - 1] = 0.5 * splineMods[gridDim - 2];

        // Invert, to avoid division later on.
        for (int i = 0; i < gridDim; ++i) splineMods[i] = 1.0 / splineMods[i];
        return splineMods;
    }

    /*!
     * \brief Gets the grid point to start interpolating from.
     * \return the index of the first grid point this spline supports.
     */
    short startingGridPoint() const { return startingGridPoint_; }

    /*!
     * \brief Returns the B-Spline, or derivative thereof.
     * \param deriv the derivative level of the spline to be returned.
     */
    const Real *operator[](short deriv) const { return splines_[deriv]; }

    /*!
     * \brief Get read-only access to the full spline data.
     * \returns a const reference to the full spline data: row index is derivative, col index is spline component.
     */
    const Matrix<Real> &splineData() const { return splines_; }
};

}  // Namespace helpme
#endif  // Header guard
