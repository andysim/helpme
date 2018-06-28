// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_GAMMA_H_
#define _HELPME_GAMMA_H_

#include <cmath>
#include <limits>

/*!
 * \file gamma.h
 * \brief Contains C++ implementations of templated gamma and incomplete gamma functions, computed using recursion.
 */

namespace helpme {

constexpr long double sqrtPi = 1.77245385090551602729816748334114518279754945612238712821381L;

/*!
 * Compute upper incomplete gamma functions for positive half-integral s values using the recursion
 * \f$ \Gamma[\frac{\mathrm{twoS}}{2},x] = \Gamma[\frac{\mathrm{twoS}-2}{2},x] + x^{\frac{\mathrm{twoS}-2}{2}}e^{-x}\f$
 */
template <typename Real, int twoS, bool isPositive>
struct incompleteGammaRecursion {
    static Real compute(Real x) {
        return (0.5f * twoS - 1) * incompleteGammaRecursion<Real, twoS - 2, isPositive>::compute(x) +
               pow(x, (0.5f * twoS - 1)) * exp(-x);
    }
};

/*!
 * Compute upper incomplete gamma functions for negative half-integral s values using the recursion
 * \f$ \Gamma[\frac{\mathrm{twoS}}{2},x] = \frac{2\Gamma[\frac{\mathrm{twoS}+2}{2},x] -
 * 2x^\frac{\mathrm{twoS}}{2}e^{-x}}{\mathrm{twoS}}\f$
 */
template <typename Real, int twoS>
struct incompleteGammaRecursion<Real, twoS, false> {
    static Real compute(Real x) {
        return (incompleteGammaRecursion<Real, twoS + 2, false>::compute(x) - pow(x, 0.5f * twoS) * exp(-x)) /
               (0.5f * twoS);
    }
};

/// Specific value of incomplete gamma function.
template <typename Real>
struct incompleteGammaRecursion<Real, 2, true> {
    static Real compute(Real x) { return exp(-x); }
};

/// Specific value of incomplete gamma function.
template <typename Real>
struct incompleteGammaRecursion<Real, 1, false> {
    static Real compute(Real x) { return sqrtPi * erfc(std::sqrt(x)); }
};

/// Specific value of incomplete gamma function.
template <typename Real>
struct incompleteGammaRecursion<Real, 1, true> {
    static Real compute(Real x) { return sqrtPi * erfc(std::sqrt(x)); }
};

/// Specific value of incomplete gamma function.
template <typename Real>
struct incompleteGammaRecursion<Real, 0, false> {
    static Real compute(Real x) {
        // Gamma(0,x) is (minus) the exponential integral of -x.  This implementation was stolen from
        // http://www.mymathlib.com/c_source/functions/exponential_integrals/exponential_integral_Ei.c
        x = -x;
        if (x < -5.0L) return -(Real)Continued_Fraction_Ei(x);
        if (x == 0.0L) return std::numeric_limits<Real>::max();
        if (x < 6.8L) return -(Real)Power_Series_Ei(x);
        if (x < 50.0L) return -(Real)Argument_Addition_Series_Ei(x);
        return -(Real)Continued_Fraction_Ei(x);
    }

   private:
    static constexpr long double epsilon = 10.0 * std::numeric_limits<long double>::epsilon();

    ////////////////////////////////////////////////////////////////////////////////
    // static long double Continued_Fraction_Ei( long double x )                  //
    //                                                                            //
    //  Description:                                                              //
    //     For x < -5 or x > 50, the continued fraction representation of Ei      //
    //     converges fairly rapidly.                                              //
    //                                                                            //
    //     The continued fraction expansion of Ei(x) is:                          //
    //        Ei(x) = -exp(x) { 1/(-x+1-) 1/(-x+3-) 4/(-x+5-) 9/(-x+7-) ... }.    //
    //                                                                            //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double  x                                                         //
    //                The argument of the exponential integral Ei().              //
    //                                                                            //
    //  Return Value:                                                             //
    //     The value of the exponential integral Ei evaluated at x.               //
    ////////////////////////////////////////////////////////////////////////////////

    static long double Continued_Fraction_Ei(long double x) {
        long double Am1 = 1.0L;
        long double A0 = 0.0L;
        long double Bm1 = 0.0L;
        long double B0 = 1.0L;
        long double a = std::exp(x);
        long double b = -x + 1.0L;
        long double Ap1 = b * A0 + a * Am1;
        long double Bp1 = b * B0 + a * Bm1;
        int j = 1;

        a = 1.0L;
        while (std::fabs(Ap1 * B0 - A0 * Bp1) > epsilon * std::fabs(A0 * Bp1)) {
            if (std::fabs(Bp1) > 1.0L) {
                Am1 = A0 / Bp1;
                A0 = Ap1 / Bp1;
                Bm1 = B0 / Bp1;
                B0 = 1.0L;
            } else {
                Am1 = A0;
                A0 = Ap1;
                Bm1 = B0;
                B0 = Bp1;
            }
            a = -j * j;
            b += 2.0L;
            Ap1 = b * A0 + a * Am1;
            Bp1 = b * B0 + a * Bm1;
            j += 1;
        }
        return (-Ap1 / Bp1);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // static long double Power_Series_Ei( long double x )                        //
    //                                                                            //
    //  Description:                                                              //
    //     For -5 < x < 6.8, the power series representation for (Ei(x) - gamma   //
    //     - ln|x|)/exp(x) is used, where gamma is Euler's gamma constant.        //
    //     Note that for x = 0.0, Ei is -inf.  In which case -DBL_MAX is          //
    //     returned.                                                              //
    //                                                                            //
    //     The power series expansion of (Ei(x) - gamma - ln|x|) / exp(x) is      //
    //        - Sum(1 + 1/2 + ... + 1/j) (-x)^j / j!, where the Sum extends       //
    //        from j = 1 to inf.                                                  //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double  x                                                         //
    //                The argument of the exponential integral Ei().              //
    //                                                                            //
    //  Return Value:                                                             //
    //     The value of the exponential integral Ei evaluated at x.               //
    ////////////////////////////////////////////////////////////////////////////////

    static long double Power_Series_Ei(long double x) {
        long double xn = -x;
        long double Sn = -x;
        long double Sm1 = 0.0L;
        long double hsum = 1.0L;
        long double g = 0.5772156649015328606065121L;
        long double y = 1.0L;
        long double factorial = 1.0L;

        while (std::fabs(Sn - Sm1) > epsilon * std::fabs(Sm1)) {
            Sm1 = Sn;
            y += 1.0L;
            xn *= (-x);
            factorial *= y;
            hsum += (1.0 / y);
            Sn += hsum * xn / factorial;
        }
        return (g + std::log(std::fabs(x)) - std::exp(x) * Sn);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // static long double Argument_Addition_Series_Ei(long double x)              //
    //                                                                            //
    //  Description:                                                              //
    //     For 6.8 < x < 50.0, the argument addition series is used to calculate  //
    //     Ei.                                                                    //
    //                                                                            //
    //     The argument addition series for Ei(x) is:                             //
    //      Ei(x+dx) = Ei(x) + exp(x) Sum j! [exp(j) expj(-dx) - 1] / x^(j+1),    //
    //     where the Sum extends from j = 0 to inf, |x| > |dx| and expj(y) is     //
    //     the exponential polynomial expj(y) = Sum y^k / k!,                     //
    //     the Sum extending from k = 0 to k = j.                                 //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double  x                                                         //
    //                The argument of the exponential integral Ei().              //
    //                                                                            //
    //  Return Value:                                                             //
    //     The value of the exponential integral Ei evaluated at x.               //
    ////////////////////////////////////////////////////////////////////////////////
    static long double Argument_Addition_Series_Ei(long double x) {
        static long double ei[] = {
            1.915047433355013959531e2L,  4.403798995348382689974e2L,  1.037878290717089587658e3L,
            2.492228976241877759138e3L,  6.071406374098611507965e3L,  1.495953266639752885229e4L,
            3.719768849068903560439e4L,  9.319251363396537129882e4L,  2.349558524907683035782e5L,
            5.955609986708370018502e5L,  1.516637894042516884433e6L,  3.877904330597443502996e6L,
            9.950907251046844760026e6L,  2.561565266405658882048e7L,  6.612718635548492136250e7L,
            1.711446713003636684975e8L,  4.439663698302712208698e8L,  1.154115391849182948287e9L,
            3.005950906525548689841e9L,  7.842940991898186370453e9L,  2.049649711988081236484e10L,
            5.364511859231469415605e10L, 1.405991957584069047340e11L, 3.689732094072741970640e11L,
            9.694555759683939661662e11L, 2.550043566357786926147e12L, 6.714640184076497558707e12L,
            1.769803724411626854310e13L, 4.669055014466159544500e13L, 1.232852079912097685431e14L,
            3.257988998672263996790e14L, 8.616388199965786544948e14L, 2.280446200301902595341e15L,
            6.039718263611241578359e15L, 1.600664914324504111070e16L, 4.244796092136850759368e16L,
            1.126348290166966760275e17L, 2.990444718632336675058e17L, 7.943916035704453771510e17L,
            2.111342388647824195000e18L, 5.614329680810343111535e18L, 1.493630213112993142255e19L,
            3.975442747903744836007e19L, 1.058563689713169096306e20L};
        int k = (int)(x + 0.5f);
        int j = 0;
        long double xx = (long double)k;
        long double dx = x - xx;
        long double xxj = xx;
        long double edx = std::exp(dx);
        long double Sm = 1.0L;
        long double Sn = (edx - 1.0L) / xxj;
        long double term = std::numeric_limits<double>::max();
        long double factorial = 1.0L;
        long double dxj = 1.0L;

        while (std::fabs(term) > epsilon * std::fabs(Sn)) {
            j++;
            factorial *= (long double)j;
            xxj *= xx;
            dxj *= (-dx);
            Sm += (dxj / factorial);
            term = (factorial * (edx * Sm - 1.0L)) / xxj;
            Sn += term;
        }

        return ei[k - 7] + Sn * std::exp(xx);
    }
};

/*!
 * Compute gamma function for positive half-integral s values using the recursion.
 * \f$ \Gamma[\frac{\mathrm{twoS}}{2}] = \Gamma[\frac{\mathrm{twoS}-2}{2}]\frac{\mathrm{twoS}-2}{2} \f$
 */
template <typename Real, int twoS, bool isPositive>
struct gammaRecursion {
    static constexpr Real value = gammaRecursion<Real, twoS - 2, isPositive>::value * (0.5f * twoS - 1);
};

/*!
 * Compute gamma function for negative half-integral s values using the recursion.
 * \f$ \Gamma[\frac{\mathrm{twoS}}{2}] = \frac{2\Gamma[\frac{\mathrm{twoS}_2}{2}]}{\mathrm{twoS}} \f$
 * Returns infinity (expressed as the largest value representable by Real) for \f$twoS = 0, -2, -4, -6, \ldots\f$ .
 */
template <typename Real, int twoS>
struct gammaRecursion<Real, twoS, false> {
    static constexpr Real value = gammaRecursion<Real, twoS + 2, false>::value == std::numeric_limits<Real>::max()
                                      ? std::numeric_limits<Real>::max()
                                      : gammaRecursion<Real, twoS + 2, false>::value / (0.5f * twoS);
};

/// Specific value of the Gamma function.
template <typename Real>
struct gammaRecursion<Real, 0, false> {
    static constexpr Real value = std::numeric_limits<Real>::max();
};

/// Specific value of the Gamma function.
template <typename Real>
struct gammaRecursion<Real, 1, true> {
    static constexpr Real value = sqrtPi;
};

/// Specific value of the Gamma function.
template <typename Real>
struct gammaRecursion<Real, 1, false> {
    static constexpr Real value = sqrtPi;
};

/// Specific value of the Gamma function.
template <typename Real>
struct gammaRecursion<Real, 2, true> {
    static constexpr Real value = 1;
};

/// Specific value of the Gamma function.
template <typename Real>
struct gammaRecursion<Real, 2, false> {
    static constexpr Real value = 1;
};

/*!
 * \class incompleteGammaComputer
 * \brief Computes the upper incomplete Gamma function.
 * \f$ \Gamma[s,x] = \int_x^\infty t^{s-1} e^{-t} \mathrm{d}t \f$
 * In this code we only need half integral arguments for \f$s\f$, and only positive \f$x\f$ arguments.
 * \tparam Real the floating point type to use for arithmetic.
 * \tparam twoS twice the s value required.
 */
template <typename Real, int twoS>
struct incompleteGammaComputer {
    /*!
     * \brief Computes the incomplete gamma function.
     * \param x value required.
     * \return \f$\Gamma[\frac{\mathrm{twoS}}{2}, x^2]\f$.
     */
    static Real compute(Real x) { return incompleteGammaRecursion<Real, twoS, (twoS > 0)>::compute(x); }
};

/*!
 * Compute upper incomplete gamma functions for positive half-integral s values using the recursion
 * \f$ \Gamma[\frac{\mathrm{twoS}}{2},x] = \Gamma[\frac{\mathrm{twoS}-2}{2},x] + x^{\frac{\mathrm{twoS}-2}{2}}e^{-x}\f$
 */
template <typename Real, int twoS, bool isPositive>
struct incompleteVirialGammaRecursion {
    static std::pair<Real, Real> compute(Real x) {
        Real gamma = incompleteGammaComputer<Real, twoS>::compute(x);
        return {gamma, (0.5f * twoS) * gamma + pow(x, (0.5f * twoS)) * exp(-x)};
    }
};

/*!
 * Compute upper incomplete gamma functions for negative half-integral s values using the recursion
 * \f$ \Gamma[\frac{\mathrm{twoS}}{2},x] = \frac{2\Gamma[\frac{\mathrm{twoS}+2}{2},x] -
 * 2x^\frac{\mathrm{twoS}}{2}e^{-x}}{\mathrm{twoS}}\f$
 */
template <typename Real, int twoS>
struct incompleteVirialGammaRecursion<Real, twoS, false> {
    static std::pair<Real, Real> compute(Real x) {
        Real gamma = incompleteGammaComputer<Real, twoS + 2>::compute(x);
        return {(gamma - pow(x, 0.5f * twoS) * exp(-x)) / (0.5f * twoS), gamma};
    }
};

/*!
 * \class incompleteGammaVirialComputer
 * \brief Computes the upper incomplete Gamma function for two different values: s and s+1.
 * \f$ \Gamma[s,x] = \int_x^\infty t^{s-1} e^{-t} \mathrm{d}t \f$
 * In this code we only need half integral arguments for \f$s\f$, and only positive \f$x\f$ arguments.
 * \tparam Real the floating point type to use for arithmetic.
 * \tparam twoS twice the s value required.
 */
template <typename Real, int twoS>
struct incompleteGammaVirialComputer {
    /*!
     * \brief Computes the incomplete gamma function for argument twoS and twoS+2.
     * \param x value required.
     * \return \f$\Gamma[\frac{\mathrm{twoS}}{2}, x]\f$ and \f$\Gamma[\frac{\mathrm{twoS+2}}{2}, x]\f$.
     */
    static std::pair<Real, Real> compute(Real x) {
        return incompleteVirialGammaRecursion<Real, twoS, (twoS >= 0)>::compute(x);
    }
};

/*!
 * \class gammaComputer
 * \brief Computes the Gamma function.
 * \f$ \Gamma[s] = \int_0^\infty t^{s-1} e^{-t} \mathrm{d}t \f$
 * In this code we only need half integral values for the \f$s\f$ argument, so the input
 * argument \f$s\f$ will yield \f$\Gamma[\frac{s}{2}]\f$.
 * \tparam Real the floating point type to use for arithmetic.
 * \tparam twoS twice the s value required.
 */
template <typename Real, int twoS>
struct gammaComputer {
    /// The value of \f$\Gamma[\frac{\mathrm{twos}}{2}]\f$
    static constexpr Real value = gammaRecursion<Real, twoS, (twoS > 0)>::value;
};

/*!
 * \brief Computes the Gamma function using recursion instead of template metaprogramming.
 * \f$ \Gamma[s] = \int_0^\infty t^{s-1} e^{-t} \mathrm{d}t \f$
 * In this code we only need half integral values for the \f$s\f$ argument, so the input
 * argument \f$s\f$ will yield \f$\Gamma[\frac{s}{2}]\f$.
 * \tparam Real the floating point type to use for arithmetic.
 * \param twoS twice the s value required.
 */
template <typename Real>
Real nonTemplateGammaComputer(int twoS) {
    if (twoS == 1) {
        return sqrtPi;
    } else if (twoS == 2) {
        return 1;
    } else if (twoS <= 0 && twoS % 2 == 0) {
        return std::numeric_limits<Real>::max();
    } else if (twoS > 0) {
        return nonTemplateGammaComputer<Real>(twoS - 2) * (0.5f * twoS - 1);
    } else {
        return nonTemplateGammaComputer<Real>(twoS + 2) / (0.5f * twoS);
    }
}

}  // Namespace helpme
#endif  // Header guard
