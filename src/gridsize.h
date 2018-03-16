// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_GRIDSIZE_H_
#define _HELPME_GRIDSIZE_H_

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <vector>

namespace helpme {

// N.B. The templates here are just to avoid multiple definitions in the .so file.

/*!
 * \brief allDivisors checks that a list of values are divisors of a given input value.
 * \param gridSize the gridSize to check for divisors.
 * \param requiredDivisors the list of divisors.
 * \return whether all listed values are divisors of gridSize.
 */
template <typename T>
bool allDivisors(T gridSize, const std::initializer_list<T> &requiredDivisors) {
    for (const T &divisor : requiredDivisors)
        if (gridSize % divisor) return false;
    return true;
}

/*!
 * \brief findGridSize FFTW likes to have transformations with dimensions of the form
 *
 *       a  b  c  d   e   f
 *      2  3  5  7  11  13
 *
 * where a,b,c and d are general and e+f is either 0 or 1. MKL has similar demands:
 *
 *   https://software.intel.com/en-us/articles/fft-length-and-layout-advisor/
 *   http://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html
 *
 * This routine will compute the next largest grid size subject to the constraint that the
 * resulting size is a multiple of a given factor.
 * \param inputSize the minimum size of the grid.
 * \param requiredDivisors list of values that must be a factor of the output grid size.
 * \return the adjusted grid size.
 */
template <typename T>
int findGridSize(T inputSize, const std::initializer_list<T> &requiredDivisors) {
    std::vector<int> primeFactors{2, 3, 5, 7};
    T minDivisor = std::min(requiredDivisors);
    T currentSize = minDivisor * std::ceil(static_cast<float>(inputSize) / minDivisor);
    while (true) {
        // Now we know that the grid size is a multiple of requiredFactor, check
        // that it satisfies the prime factor requirements stated above.
        T remainder = currentSize;
        for (const int &factor : primeFactors)
            while (remainder > 1 && remainder % factor == 0) remainder /= factor;
        if ((remainder == 1 || remainder == 11 || remainder == 13) && allDivisors(currentSize, requiredDivisors))
            return currentSize;
        currentSize += minDivisor;
    }
}

}  // Namespace helpme

#endif  // Header guard
