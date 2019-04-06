// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#ifndef _HELPME_FLETCHER_H_
#define _HELPME_FLETCHER_H_

namespace helpme {
/*!
 * \brief compute_fletcher_checksum16 computes the 16bit Fletcher checksum on some
 *        data.  Only the data that fit exactly into 8bit chunks are considered.
 * \param data pointer to the data to checksum.
 * \param nElements the number of elements in the array to checksum.
 * \return the Fletcher checksum.
 */
template <typename Real>
uint16_t compute_fletcher_checksum16(const Real *data, size_t nElements) {
    size_t len = nElements * sizeof(Real) / 1;
    if (len == 0) throw std::runtime_error("Not enough data to compute a fletcher16 checksum");
    const auto *ptr = reinterpret_cast<const uint8_t *>(data);
    uint16_t sum1 = 0, sum2 = 0;
    const size_t skip = 8;
    const uint16_t modulo = (1L << skip) - 1;
    while (len--) {
        sum1 += *ptr++;
        sum1 -= sum1 >= modulo ? modulo : 0;
        sum2 += sum1;
        sum2 -= sum2 >= modulo ? modulo : 0;
    }
    return (sum2 << skip | sum1);
}

/*!
 * \brief compute_fletcher_checksum32 computes the 32bit Fletcher checksum on some
 *        data.  Only the data that fit exactly into 16bit chunks are considered.
 * \param data pointer to the data to checksum.
 * \param nElements the number of elements in the array to checksum.
 * \return the Fletcher checksum.
 */
template <typename Real>
uint32_t compute_fletcher_checksum32(const Real *data, size_t nElements) {
    size_t len = nElements * sizeof(Real) / 2;
    if (len == 0) throw std::runtime_error("Not enough data to compute a fletcher32 checksum");
    const auto *ptr = reinterpret_cast<const uint16_t *>(data);
    uint32_t sum1 = 0, sum2 = 0;
    const size_t skip = 16;
    const uint32_t modulo = (1L << skip) - 1;
    while (len--) {
        sum1 += *ptr++;
        sum1 -= sum1 >= modulo ? modulo : 0;
        sum2 += sum1;
        sum2 -= sum2 >= modulo ? modulo : 0;
    }
    return (sum2 << skip | sum1);
}

/*!
 * \brief compute_fletcher_checksum64 computes the 64bit Fletcher checksum on some
 *        data.  Only the data that fit exactly into 32bit chunks are considered.
 * \param data pointer to the data to checksum.
 * \param nElements the number of elements in the array to checksum.
 * \return the Fletcher checksum.
 */
template <typename Real>
uint64_t compute_fletcher_checksum64(const Real *data, size_t nElements) {
    size_t len = nElements * sizeof(Real) / 4;
    if (len == 0) throw std::runtime_error("Not enough data to compute a fletcher64 checksum");
    const auto *ptr = reinterpret_cast<const uint32_t *>(data);
    uint64_t sum1 = 0, sum2 = 0;
    const size_t skip = 32;
    const uint64_t modulo = (1L << skip) - 1;
    while (len--) {
        sum1 += *ptr++;
        sum1 -= sum1 >= modulo ? modulo : 0;
        sum2 += sum1;
        sum2 -= sum2 >= modulo ? modulo : 0;
    }
    return (sum2 << skip | sum1);
}
}  // namespace helpme
#endif  //_Header guard
