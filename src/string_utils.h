// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_STRING_UTIL_H_
#define _HELPME_STRING_UTIL_H_

#include <complex>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace helpme {

/*!
 * \brief makes a string representation of a floating point number.
 * \param width the width used to display the number.
 * \param precision the precision used to display the number.
 * \return the string representation of the floating point number.
 */
template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
std::string formatNumber(const T &number, int width, int precision) {
    std::stringstream stream;
    stream.setf(std::ios::fixed, std::ios::floatfield);
    stream << std::setw(width) << std::setprecision(precision) << number;
    return stream.str();
}

/*!
 * \brief makes a string representation of a complex number.
 * \param width the width used to display the real and the imaginary components.
 * \param precision the precision used to display the real and the imaginary components.
 * \return the string representation of the complex number.
 */
template <typename T, typename std::enable_if<!std::is_floating_point<T>::value, int>::type = 0>
std::string formatNumber(const T &number, int width, int precision) {
    std::stringstream stream;
    stream.setf(std::ios::fixed, std::ios::floatfield);
    stream << "(" << std::setw(width) << std::setprecision(precision) << number.real() << ", " << std::setw(width)
           << std::setprecision(precision) << number.imag() << ")";
    return stream.str();
}

/*!
 * \brief makes a string representation of a multdimensional tensor, stored in a flat array.
 * \param data pointer to the start of the array holding the tensor information.
 * \param size the length of the array holding the tensor information.
 * \param rowDim the dimension of the fastest running index.
 * \param width the width of each individual floating point number.
 * \param precision used to display each floating point number.
 * \return the string representation of the tensor.
 */
template <typename T>
std::string stringify(T *data, size_t size, size_t rowDim, int width = 14, int precision = 8) {
    std::stringstream stream;
    for (size_t ind = 0; ind < size; ++ind) {
        stream << formatNumber(data[ind], width, precision);
        if (ind % rowDim == rowDim - 1)
            stream << std::endl;
        else
            stream << "  ";
    }
    return stream.str();
}

}  // Namespace helpme

#endif  // Header guard
