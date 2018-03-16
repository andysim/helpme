// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_POWERS_H_
#define _HELPME_POWERS_H_

#include <cmath>

/*!
 * \file powers.h
 * \brief Contains template functions to compute various quantities raised to an integer power.
 */

namespace helpme {

template <typename Real, int n>
struct raiseToIntegerPower {
    static Real pow(Real val) { return val * raiseToIntegerPower<Real, n - 1>::pow(val); }
};

/// Base recursion for the power.
template <typename Real>
struct raiseToIntegerPower<Real, 0> {
    static Real pow(Real) { return 1; }
};

/// n is positive and even case
template <typename Real, int n, bool nIsPositive, bool nIsEven>
struct normIntegerPowerComputer {
    static Real compute(Real val) { return raiseToIntegerPower<Real, n / 2>::pow(val); }
};

/// n is positive and odd case
template <typename Real, int n>
struct normIntegerPowerComputer<Real, n, true, false> {
    static Real compute(Real val) { return raiseToIntegerPower<Real, n>::pow(std::sqrt(val)); }
};

/// n is negative and even case
template <typename Real, int n>
struct normIntegerPowerComputer<Real, n, false, true> {
    static Real compute(Real val) { return raiseToIntegerPower<Real, -n / 2>::pow(1 / val); }
};

/// n is negative and odd case
template <typename Real, int n>
struct normIntegerPowerComputer<Real, n, false, false> {
    static Real compute(Real val) { return raiseToIntegerPower<Real, -n>::pow(1 / sqrt(val)); }
};

/*!
 * \brief Compute a quantity exponentiated by an integer power, using multiplication,
 * at compile time.  The exponent is assumed to be positve.
 * \tparam Real the floating point type to use for arithmetic.
 * \tparam n the exponent to raise the value to.
 */
template <typename Real, int n>
struct raiseNormToIntegerPower {
    /*!
     * \brief pow compute the norm raised to the power n.
     * \param val the square of the norm to be exponentiated.
     * \return the norm raised to the integer power.
     */
    static Real compute(Real val) { return normIntegerPowerComputer<Real, n, (n >= 0), (n % 2 == 0)>::compute(val); }
};
}  // Namespace helpme

#endif  // Header guard
