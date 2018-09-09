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
    inline void makeSplineInPlace(Real *array, const Real &val, const short &n) const {
        Real denom = (Real)1 / (n - 1);
        array[n - 1] = denom * val * array[n - 2];
        for (short j = 1; j < n - 1; ++j)
            array[n - j - 1] = denom * ((val + j) * array[n - j - 2] + (n - j - val) * array[n - j - 1]);
        array[0] *= denom * (1 - val);
    }

    /// Takes BSpline derivative.
    inline void differentiateSpline(const Real *array, Real *dArray, const short &n) const {
        dArray[0] = -array[0];
        for (short j = 1; j < n - 1; ++j) dArray[j] = array[j - 1] - array[j];
        dArray[n - 1] = array[n - 2];
    }

    /*!
     * \brief assertSplineIsSufficient ensures that the spline is large enough to be differentiable.
     *        An mth order B-Spline is differentiable m-2 times.
     */
    void assertSplineIsSufficient(int splineOrder, int derivativeLevel) const {
        if (splineOrder - derivativeLevel < 2) {
            std::string msg(
                "The spline order used is not sufficient for the derivative level requested."
                "Set the spline order to at least ");
            msg += std::to_string(derivativeLevel + 2);
            msg += " to run this calculation.";
            throw std::runtime_error(msg);
        }
    }

   public:
    /// The B-splines and their derivatives.  See update() for argument details.
    BSpline(short start, Real value, short order, short derivativeLevel) : splines_(derivativeLevel + 1, order) {
        update(start, value, order, derivativeLevel);
    }

    /*!
     * \brief update computes information for BSpline, without reallocating memory unless needed.
     * \param start the grid point at which to start interpolation.
     * \param value the distance (in fractional coordinates) from the starting grid point.
     * \param order the order of the BSpline.
     * \param derivativeLevel the maximum level of derivative needed for this BSpline.
     */
    void update(short start, Real value, short order, short derivativeLevel) {
        assertSplineIsSufficient(order, derivativeLevel);
        startingGridPoint_ = start;
        order_ = order;
        derivativeLevel_ = derivativeLevel;

        // The +1 is to account for the fact that we need to store entries up to and including the max.
        if (splines_.nRows() < derivativeLevel + 1 || splines_.nCols() != order)
            splines_ = Matrix<Real>(derivativeLevel + 1, order);

        splines_.setZero();
        splines_(0, 0) = 1 - value;
        splines_(0, 1) = value;
        for (short m = 1; m < order_ - 1; ++m) {
            makeSplineInPlace(splines_[0], value, m + 2);
            if (m >= order_ - derivativeLevel_ - 2) {
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
     * \param mValues if provided, provides the ordering of the m values, if not they are
     *        ordered as 0, 1, 2, ..., Kmax, -Kmax+1, -Kmax+2, ..., -2, -1.
     * \return a gridDim long vector containing the inverse of the Fourier space spline moduli.
     */
    helpme::vector<Real> invSplineModuli(short gridDim, std::vector<int> mValues = {}) {
        int nKTerms = mValues.size() ? mValues.size() : gridDim;
        helpme::vector<Real> splineMods(nKTerms, 0);
        Real prefac = 2 * M_PI / gridDim;
        for (int m = 0; m < nKTerms; ++m) {
            Real real = 0;
            Real imag = 0;
            int mValue = mValues.size() ? mValues[m] : m;
            for (int n = 0; n < order_; ++n) {
                Real exparg = mValue * n * prefac;
                Real jSpline = splines_(0, n);
                real += jSpline * cos(exparg);
                imag += jSpline * sin(exparg);
            }
            splineMods[m] = real * real + imag * imag;
        }

        // Correct tiny values for conventional PME.
        if (!mValues.size()) {
            constexpr Real EPS = 1e-7f;
            if (splineMods[0] < EPS) splineMods[0] = splineMods[1] / 2;
            for (int i = 0; i < gridDim - 1; ++i)
                if (splineMods[i] < EPS) splineMods[i] = (splineMods[i - 1] + splineMods[i + 1]) / 2;
            if (splineMods[gridDim - 1] < EPS) splineMods[gridDim - 1] = splineMods[gridDim - 2] / 2;
        }

        // Invert, to avoid division later on.
        for (int i = 0; i < nKTerms; ++i) splineMods[i] = 1 / splineMods[i];

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
    const Real *operator[](const int &deriv) const { return splines_[deriv]; }

    /*!
     * \brief Get read-only access to the full spline data.
     * \returns a const reference to the full spline data: row index is derivative, col index is spline component.
     */
    const Matrix<Real> &splineData() const { return splines_; }
};

}  // Namespace helpme
#endif  // Header guard
