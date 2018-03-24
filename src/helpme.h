// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_HELPME_H_
#define _HELPME_HELPME_H_

#if __cplusplus || DOXYGEN

// C++ header

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

#include "fftw_wrapper.h"
#include "gamma.h"
#include "gridsize.h"
#include "matrix.h"
#include "memory.h"
#if HAVE_MPI == 1
#include "mpi_wrapper.h"
#else
typedef struct ompi_communicator_t *MPI_Comm;
#endif
#include "powers.h"
#include "splines.h"

/*!
 * \file helpme.h
 * \brief Contains the C++ implementation of a PME Instance, and related helper classes.
 */

namespace helpme {

/*!
 * \brief nCartesian computes the total number of Cartesian components of a given angular momentum.
 * \param L the angular momentum.
 * \return total number of components up to and including angular momentum L.
 */
static int nCartesian(int L) { return (L + 1) * (L + 2) * (L + 3) / 6; }

/*!
 * \brief cartAddress computes the address of a term with given quantum numbers in a Cartesian buffer.
 * \param lx the x quantum number.
 * \param ly the y quantum number.
 * \param lz the z quantum number.
 * \return the address of an {lx, ly, lz} quantity in a buffer that contains all lower angular momentum terms too.
 */
static int cartAddress(int lx, int ly, int lz) {
    int l = lx + ly + lz;
    return l * (l + 1) * (l + 2) / 6 + lz * (l * 2 - lz + 3) / 2 + ly;
}

// This is used to define function pointers in the constructor, and makes it easy to add new kernels.
#define ENABLE_KERNEL_WITH_INVERSE_R_EXPONENT_OF(n) \
    case n:                                         \
        convolveEFxn_ = &convolveEImpl<n>;          \
        convolveEVFxn_ = &convolveEVImpl<n>;        \
        slfEFxn_ = &slfEImpl<n>;                    \
        dirEFxn_ = &dirEImpl<n>;                    \
        adjEFxn_ = &adjEImpl<n>;                    \
        dirEFFxn_ = &dirEFImpl<n>;                  \
        adjEFFxn_ = &adjEFImpl<n>;                  \
        break;

/*!
 * \class PMEInstance
 * \brief A class to encapsulate information related to a particle mesh Ewald calculation.
 *
 * By storing information related to a single PME calculation in this way, we allow multiple
 * instances to be created in calculations requiring multiple PMEs, e.g. for computing both
 * electrostatic and attractive dispersion terms using PME to handle long-range interactions.
 * \tparam Real the floating point type to use for arithmetic.
 */
template <typename Real>
class PMEInstance {
    using GridIterator = std::vector<std::vector<std::pair<short, short>>>;
    using Complex = std::complex<Real>;
    using RealMat = Matrix<Real>;
    using RealVec = helpme::vector<Real>;

   protected:
    /// The FFT grid dimensions in the {A,B,C} grid dimensions.
    int dimA_, dimB_, dimC_;
    /// The full A dimension after real->complex transformation.
    int complexDimA_;
    /// The locally owned A dimension after real->complex transformation.
    int myComplexDimA_;
    /// The order of the cardinal B-Spline used for interpolation.
    int splineOrder_;
    /// The number of threads per MPI instance.
    int nThreads_;
    /// The scale factor to apply to all energies and derivatives.
    Real scaleFactor_;
    /// The attenuation parameter, whose units should be the inverse of those used to specify coordinates.
    Real kappa_;
    /// The lattice vectors.
    RealMat boxVecs_;
    /// The reciprocal lattice vectors.
    RealMat recVecs_;
    /// The scaled reciprocal lattice vectors, for transforming forces from scaled fractional coordinates.
    RealMat scaledRecVecs_;
    /// An iterator over angular momentum components.
    std::vector<std::array<short, 3>> angMomIterator_;
    /// The number of permutations of each multipole component.
    RealVec permutations_;
    /// From a given starting point on the {A,B,C} edge of the grid, lists all points to be handled, correctly wrapping
    /// around the end.
    GridIterator gridIteratorA_, gridIteratorB_, gridIteratorC_;
    /// The (inverse) bspline moduli to normalize the spreading / probing steps; these are folded into the convolution.
    RealVec splineModA_, splineModB_, splineModC_;
    /// A function pointer to call the approprate function to implement convolution, templated to the rPower value.
    std::function<Real(int, int, int, int, int, int, int, Real, Complex *, const RealMat &, Real, Real, const RealVec &,
                       const RealVec &, const RealVec &)>
        convolveEFxn_;
    /// A function pointer to call the approprate function to implement convolution with virial, templated to
    /// the rPower value.
    std::function<Real(int, int, int, int, int, int, int, Real, Complex *, const RealMat &, Real, Real, const RealVec &,
                       const RealVec &, const RealVec &, RealMat &)>
        convolveEVFxn_;
    /// A function pointer to call the approprate function to compute self energy, templated to the rPower value.
    std::function<Real(int, const RealMat &, Real, Real)> slfEFxn_;
    /// A function pointer to call the approprate function to compute the direct energy, templated to the rPower value.
    std::function<Real(Real, Real)> dirEFxn_;
    /// A function pointer to call the approprate function to compute the adjusted energy, templated to the rPower
    /// value.
    std::function<Real(Real, Real)> adjEFxn_;
    /// A function pointer to call the approprate function to compute the direct energy and force, templated to the
    /// rPower value.
    std::function<std::tuple<Real, Real>(Real, Real, Real)> dirEFFxn_;
    /// A function pointer to call the approprate function to compute the adjusted energy and force, templated to the
    /// rPower value.
    std::function<std::tuple<Real, Real>(Real, Real, Real)> adjEFFxn_;
#if HAVE_MPI == 1
    /// The communicator object that handles interactions with MPI.
    std::unique_ptr<MPIWrapper<Real>> mpiCommunicator_;
    /// The communicator object that handles interactions with MPI along this nodes {A,B,C} pencils.
    std::unique_ptr<MPIWrapper<Real>> mpiCommunicatorA_, mpiCommunicatorB_, mpiCommunicatorC_;
#endif
    /// The number of nodes in the {A,B,C} dimensions.
    int numNodesA_, numNodesB_, numNodesC_;
    /// The rank of this node along the {A,B,C} dimensions.
    int rankA_, rankB_, rankC_;
    /// The first grid point that this node is responsible for in the {A,B,C} dimensions.
    int firstA_, firstB_, firstC_;
    /// The grid point beyond the last point that this this node is responsible for in the {A,B,C} dimensions.
    int lastA_, lastB_, lastC_;
    /// The {X,Y,Z} dimensions of the locally owned chunk of the grid.
    int myDimA_, myDimB_, myDimC_;
    /// The subsets of a given dimension to be processed when doing a transform along another dimension.
    int subsetOfCAlongA_, subsetOfCAlongB_, subsetOfCAlongC_, subsetOfBAlongC_;
    /// Communication buffers for MPI parallelism.
    helpme::vector<Complex> workSpace1_, workSpace2_;
    /// FFTW wrappers to help with transformations in the {A,B,C} dimensions.
    FFTWWrapper<Real> fftHelperA_, fftHelperB_, fftHelperC_;

    /*!
     * \brief A simple helper to compute factorials.
     * \param n the number whose factorial is to be taken.
     * \return n!
     */
    unsigned int factorial(unsigned int n) {
        unsigned int ret = 1;
        for (unsigned int i = 1; i <= n; ++i) ret *= i;
        return ret;
    }

    /*! Make sure that the iterator over AM components is up to date.
     * \param angMom the angular momentum required for the iterator over multipole components.
     */
    void updateAngMomIterator(int parameterAngMom) {
        auto L = parameterAngMom;
        size_t expectedNTerms = nCartesian(L);
        if (angMomIterator_.size() >= expectedNTerms) return;

        angMomIterator_.resize(expectedNTerms);
        permutations_.resize(expectedNTerms);
        for (short l = 0, count = 0; l <= L; ++l) {
            for (short lz = 0; lz <= l; ++lz) {
                for (short ly = 0; ly <= l - lz; ++ly) {
                    short lx = l - ly - lz;
                    angMomIterator_[count] = {{static_cast<short>(lx), static_cast<short>(ly), static_cast<short>(lz)}};
                    permutations_[count] = (Real)factorial(l) / (factorial(lx) * factorial(ly) * factorial(lz));
                    ++count;
                }
            }
        }
    }

    /*!
     * \brief makes a string representation of a multdimensional tensor, stored in a vector.
     * \param data the vector holding the tensor information.
     * \param rowDim the dimension of the fastest running index.
     * \return the string representation of the tensor.
     */
    template <typename T>
    std::string stringify(helpme::vector<T> data, size_t rowDim) {
        std::stringstream stream;
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        for (size_t ind = 0; ind < data.size(); ++ind) {
            stream << std::setw(18) << std::setprecision(10) << data[ind];
            if (ind % rowDim == rowDim - 1)
                stream << std::endl;
            else
                stream << "  ";
        }
        return stream.str();
    }

    /*!
     * \brief makes a string representation of a multdimensional tensor, stored in a vector.
     * \param data the array holding the tensor information.
     * \param size the length of the array holding the tensor information.
     * \param rowDim the dimension of the fastest running index.
     * \return the string representation of the tensor.
     */
    template <typename T>
    std::string stringify(T *data, size_t size, size_t rowDim) {
        std::stringstream stream;
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        for (size_t ind = 0; ind < size; ++ind) {
            stream << std::setw(18) << std::setprecision(10) << data[ind];
            if (ind % rowDim == rowDim - 1)
                stream << std::endl;
            else
                stream << "  ";
        }
        return stream.str();
    }

    /*!
     * \brief makeBSplines construct the {x,y,z} B-Splines.
     * \param atomCoords a 3-vector containing the atom's coordinates.
     * \param derivativeLevel level of derivative needed for the splines.
     * \return a 3-tuple containing the {x,y,z} B-splines.
     */
    std::tuple<BSpline<Real>, BSpline<Real>, BSpline<Real>> makeBSplines(const Real *atomCoords,
                                                                         short derivativeLevel) {
        // Subtract a tiny amount to make sure we're not exactly on the rightmost (excluded)
        // grid point. The calculation is translationally invariant, so this is valid.
        constexpr float EPS = 1e-6;
        Real aCoord =
            atomCoords[0] * recVecs_(0, 0) + atomCoords[1] * recVecs_(1, 0) + atomCoords[2] * recVecs_(2, 0) - EPS;
        Real bCoord =
            atomCoords[0] * recVecs_(0, 1) + atomCoords[1] * recVecs_(1, 1) + atomCoords[2] * recVecs_(2, 1) - EPS;
        Real cCoord =
            atomCoords[0] * recVecs_(0, 2) + atomCoords[1] * recVecs_(1, 2) + atomCoords[2] * recVecs_(2, 2) - EPS;
        // Make sure the fractional coordinates fall in the range 0 <= s < 1
        aCoord -= floor(aCoord);
        bCoord -= floor(bCoord);
        cCoord -= floor(cCoord);
        short aStartingGridPoint = dimA_ * aCoord;
        short bStartingGridPoint = dimB_ * bCoord;
        short cStartingGridPoint = dimC_ * cCoord;
        Real aDistanceFromGridPoint = dimA_ * aCoord - aStartingGridPoint;
        Real bDistanceFromGridPoint = dimB_ * bCoord - bStartingGridPoint;
        Real cDistanceFromGridPoint = dimC_ * cCoord - cStartingGridPoint;
        return std::make_tuple(
            BSpline<Real>(aStartingGridPoint, aDistanceFromGridPoint, splineOrder_, derivativeLevel),
            BSpline<Real>(bStartingGridPoint, bDistanceFromGridPoint, splineOrder_, derivativeLevel),
            BSpline<Real>(cStartingGridPoint, cDistanceFromGridPoint, splineOrder_, derivativeLevel));
    }

    /*!
     * \brief sanityChecks just makes sure that inputs have consistent dimensions, and that prerequisites are
     * initialized.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.).
     * \param parameters the input parameters.
     * \param coordinates the input coordinates.
     */
    void sanityChecks(int parameterAngMom, const RealMat &parameters, const RealMat &coordinates) {
        // Start with some sanity checks.
        if (parameters.nRows() == 0)
            throw std::runtime_error("Parameters have not been set yet!  Call setParameters(...) before runPME(...);");
        if (coordinates.nRows() == 0)
            throw std::runtime_error(
                "Coordinates have not been set yet!  Call setCoordinates(...) before runPME(...);");
        if (boxVecs_.isNearZero())
            throw std::runtime_error(
                "Lattice vectors have not been set yet!  Call setLatticeVectors(...) before runPME(...);");
        if (coordinates.nRows() != parameters.nRows())
            throw std::runtime_error(
                "Inconsistent number of coordinates and parameters; there should be nAtoms of each.");
        if (parameters.nCols() != nCartesian(parameterAngMom))
            throw std::runtime_error(
                "Mismatch in the number of parameters provided and the parameter angular momentum");
    }

    /*!
     * \brief convolveEImpl performs the reciprocal space convolution, returning the energy
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param nx the grid dimension in the x direction.
     * \param ny the grid dimension in the y direction.
     * \param nz the grid dimension in the z direction.
     * \param myNx the subset of the grid in the x direction to be handled by this node.
     * \param myNy the subset of the grid in the y direction to be handled by this node.
     * \param startX the starting grid point handled by this node in the X direction.
     * \param startY the starting grid point handled by this node in the Y direction.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param gridPtr the Fourier space grid, with ordering YXZ.
     * \param boxInv the reciprocal lattice vectors.
     * \param volume the volume of the unit cell.
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param xMods the Fourier space norms of the x B-Splines.
     * \param yMods the Fourier space norms of the y B-Splines.
     * \param zMods the Fourier space norms of the z B-Splines.
     * \return the reciprocal space energy.
     */
    template <int rPower>
    static Real convolveEImpl(int nx, int ny, int nz, int myNx, int myNy, int startX, int startY, Real scaleFactor,
                              Complex *gridPtr, const RealMat &boxInv, Real volume, Real kappa, const RealVec &xMods,
                              const RealVec &yMods, const RealVec &zMods) {
        Real energy = 0;

        bool nodeZero = startX == 0 && startY == 0;
        if (rPower > 3 && nodeZero) {
            // Kernels with rPower>3 are absolutely convergent and should have the m=0 term present.
            // To compute it we need sum_ij c(i)c(j), which can be obtained from the structure factor norm.
            Real prefac = 2 * scaleFactor * M_PI * sqrtPi * pow(kappa, rPower - 3) /
                          ((rPower - 3) * gammaComputer<Real, rPower>::value * volume);
            energy += prefac * std::norm(gridPtr[0]);
        }

        // The parallel scheme leads to a very confusing situation; dividing the aDim/2+1 elements among the x Nodes
        // is achieved by assigning aDim/(2 numNodesX) values to each node.  Node 0 then has one extra element than
        // the remaining nodes, which have a slice of zeros in place of this extra element.
        int actualNx = startX ? myNx - 1 : myNx;

        std::vector<short> xMVals(actualNx), yMVals(myNy), zMVals(nz);
        // Iterators to conveniently map {X,Y,Z} grid location to m_{X,Y,Z} value, where -1/2 << m/dim < 1/2.
        for (int kx = 0; kx < actualNx; ++kx) xMVals[kx] = startX + (kx + startX >= (nx + 1) / 2 ? kx - nx : kx);
        for (int ky = 0; ky < myNy; ++ky) yMVals[ky] = startY + (ky + startY >= (ny + 1) / 2 ? ky - ny : ky);
        for (int kz = 0; kz < nz; ++kz) zMVals[kz] = kz >= (nz + 1) / 2 ? kz - nz : kz;

        Real bPrefac = M_PI * M_PI / (kappa * kappa);
        Real volPrefac = scaleFactor * pow(M_PI, rPower - 1) / (sqrtPi * gammaComputer<Real, rPower>::value * volume);
        int halfNx = nx / 2 + 1;
        size_t nxz = actualNx * nz;
        for (int ky = 0; ky < myNy; ++ky) {
            // Exclude m=0 cell.
            size_t start = ky == 0 && nodeZero ? 1 : 0;
            Complex *yPtr = gridPtr + ky * myNx * nz;
            Real my = (Real)yMVals[ky];
            Real yMod = yMods[ky + startY];
            for (size_t xz = start; xz < nxz; ++xz) {
                int kx = xz / nz;
                int kz = xz % nz;
                // We only loop over the first nx/2+1 x values; this accounts for the "missing" complex conjugate
                // values.
                Real permPrefac = kx + startX != 0 && kx + startX != halfNx - 1 ? 2 : 1;
                Real mx = (Real)xMVals[kx];
                Real mz = (Real)zMVals[kz];
                // TODO clean this up and move stuff up into outer loops.
                Real mVecX = boxInv(0, 0) * mx + boxInv(0, 1) * my + boxInv(0, 2) * mz;
                Real mVecY = boxInv(1, 0) * mx + boxInv(1, 1) * my + boxInv(1, 2) * mz;
                Real mVecZ = boxInv(2, 0) * mx + boxInv(2, 1) * my + boxInv(2, 2) * mz;
                Real mNormSq = mVecX * mVecX + mVecY * mVecY + mVecZ * mVecZ;
                Real mTerm = raiseNormToIntegerPower<Real, rPower - 3>::compute(mNormSq);
                Real bSquared = bPrefac * mNormSq;
                Real incompleteGammaTerm = incompleteGammaComputer<Real, 3 - rPower>::compute(bSquared);
                Complex &gridVal = yPtr[kx * nz + kz];
                Real structFacNorm = std::norm(gridVal);
                Real influenceFunction =
                    volPrefac * incompleteGammaTerm * mTerm * yMod * xMods[kx + startX] * zMods[kz];
                gridVal *= influenceFunction;
                energy += permPrefac * influenceFunction * structFacNorm;
            }
        }
        energy /= 2;

        return energy;
    }

    /*!
     * \brief convolveEVImpl performs the reciprocal space convolution, returning the energy
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param nx the grid dimension in the x direction.
     * \param ny the grid dimension in the y direction.
     * \param nz the grid dimension in the z direction.
     * \param myNx the subset of the grid in the x direction to be handled by this node.
     * \param myNy the subset of the grid in the y direction to be handled by this node.
     * \param startX the starting grid point handled by this node in the X direction.
     * \param startY the starting grid point handled by this node in the Y direction.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param gridPtr the Fourier space grid, with ordering YXZ.
     * \param boxInv the reciprocal lattice vectors.
     * \param volume the volume of the unit cell.
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param xMods the Fourier space norms of the x B-Splines.
     * \param yMods the Fourier space norms of the y B-Splines.
     * \param zMods the Fourier space norms of the z B-Splines.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \return the reciprocal space energy.
     */
    template <int rPower>
    static Real convolveEVImpl(int nx, int ny, int nz, int myNx, int myNy, int startX, int startY, Real scaleFactor,
                               Complex *gridPtr, const RealMat &boxInv, Real volume, Real kappa, const RealVec &xMods,
                               const RealVec &yMods, const RealVec &zMods, RealMat &virial) {
        Real energy = 0;

        bool nodeZero = startX == 0 && startY == 0;
        if (rPower > 3 && nodeZero) {
            // Kernels with rPower>3 are absolutely convergent and should have the m=0 term present.
            // To compute it we need sum_ij c(i)c(j), which can be obtained from the structure factor norm.
            Real prefac = 2 * scaleFactor * M_PI * sqrtPi * pow(kappa, rPower - 3) /
                          ((rPower - 3) * gammaComputer<Real, rPower>::value * volume);
            energy += prefac * std::norm(gridPtr[0]);
        }

        // The parallel scheme leads to a very confusing situation; dividing the aDim/2+1 elements among the x Nodes
        // is achieved by assigning aDim/(2 numNodesX) values to each node.  Node 0 then has one extra element than
        // the remaining nodes, which have a slice of zeros in place of this extra element.
        int actualNx = startX ? myNx - 1 : myNx;

        std::vector<short> xMVals(actualNx), yMVals(myNy), zMVals(nz);
        // Iterators to conveniently map {X,Y,Z} grid location to m_{X,Y,Z} value, where -1/2 << m/dim < 1/2.
        for (int kx = 0; kx < actualNx; ++kx) xMVals[kx] = startX + (kx + startX >= (nx + 1) / 2 ? kx - nx : kx);
        for (int ky = 0; ky < myNy; ++ky) yMVals[ky] = startY + (ky + startY >= (ny + 1) / 2 ? ky - ny : ky);
        for (int kz = 0; kz < nz; ++kz) zMVals[kz] = kz >= (nz + 1) / 2 ? kz - nz : kz;

        Real bPrefac = M_PI * M_PI / (kappa * kappa);
        Real volPrefac = scaleFactor * pow(M_PI, rPower - 1) / (sqrtPi * gammaComputer<Real, rPower>::value * volume);
        int halfNx = nx / 2 + 1;
        size_t nxz = actualNx * nz;
        Real Vxx = 0;
        Real Vxy = 0;
        Real Vyy = 0;
        Real Vxz = 0;
        Real Vyz = 0;
        Real Vzz = 0;
        for (int ky = 0; ky < myNy; ++ky) {
            // Exclude m=0 cell.
            size_t start = ky == 0 && nodeZero ? 1 : 0;
            Complex *yPtr = gridPtr + ky * myNx * nz;
            Real my = (Real)yMVals[ky];
            Real yMod = yMods[ky + startY];
            for (size_t xz = start; xz < nxz; ++xz) {
                int kx = xz / nz;
                int kz = xz % nz;
                // We only loop over the first nx/2+1 x values; this accounts for the "missing" complex conjugate
                // values.
                Real permPrefac = kx + startX != 0 && kx + startX != halfNx - 1 ? 2 : 1;
                Real mx = (Real)xMVals[kx];
                Real mz = (Real)zMVals[kz];
                Real mVecX = boxInv(0, 0) * mx + boxInv(0, 1) * my + boxInv(0, 2) * mz;
                Real mVecY = boxInv(1, 0) * mx + boxInv(1, 1) * my + boxInv(1, 2) * mz;
                Real mVecZ = boxInv(2, 0) * mx + boxInv(2, 1) * my + boxInv(2, 2) * mz;
                Real mNormSq = mVecX * mVecX + mVecY * mVecY + mVecZ * mVecZ;
                Real mTerm = raiseNormToIntegerPower<Real, rPower - 3>::compute(mNormSq);
                Real bSquared = bPrefac * mNormSq;
                auto gammas = incompleteGammaVirialComputer<Real, 3 - rPower>::compute(bSquared);
                Real eGamma = std::get<0>(gammas);
                Real vGamma = std::get<1>(gammas);
                Complex &gridVal = yPtr[kx * nz + kz];
                Real structFacNorm = std::norm(gridVal);
                Real totalPrefac = volPrefac * mTerm * yMod * xMods[kx + startX] * zMods[kz];
                Real influenceFunction = totalPrefac * eGamma;
                gridVal *= influenceFunction;
                Real eTerm = permPrefac * influenceFunction * structFacNorm;
                Real vTerm = permPrefac * vGamma * totalPrefac / mNormSq * structFacNorm;
                energy += eTerm;
                Vxx += vTerm * mVecX * mVecX;
                Vxy += vTerm * mVecX * mVecY;
                Vyy += vTerm * mVecY * mVecY;
                Vxz += vTerm * mVecX * mVecZ;
                Vyz += vTerm * mVecY * mVecZ;
                Vzz += vTerm * mVecZ * mVecZ;
            }
        }
        energy /= 2;

        virial[0][0] -= Vxx - energy;
        virial[0][1] -= Vxy;
        virial[0][2] -= Vyy - energy;
        virial[0][3] -= Vxz;
        virial[0][4] -= Vyz;
        virial[0][5] -= Vzz - energy;

        return energy;
    }

    /*!
     * \brief dirEImpl computes the kernel for the direct energy for a pair.
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param rSquared the square of the internuclear distance
     * \param kappaSquared the square of attenuation parameter in units inverse of those used to specify coordinates.
     * \return the energy kernel.
     */
    template <int rPower>
    inline static Real dirEImpl(Real rSquared, Real kappaSquared) {
        Real denominator = raiseNormToIntegerPower<Real, rPower>::compute(rSquared);
        Real gammaTerm = incompleteGammaComputer<Real, rPower>::compute(rSquared * kappaSquared) /
                         gammaComputer<Real, rPower>::value;
        return gammaTerm / denominator;
    }

    /*!
     * \brief dirEFImpl computes the kernels for the direct energy and force for a pair.
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param rSquared the square of the internuclear distance
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param kappaSquared the square of attenuation parameter in units inverse of those used to specify coordinates.
     * \return a tuple containing the energy and force kernels, respectively.
     */
    template <int rPower>
    inline static std::tuple<Real, Real> dirEFImpl(Real rSquared, Real kappa, Real kappaSquared) {
        Real rInv = 1 / rSquared;
        Real kappaToRPower = kappa;
        for (int i = 1; i < rPower; ++i) kappaToRPower *= kappa;
        Real denominator = raiseNormToIntegerPower<Real, rPower>::compute(rSquared);
        Real gammaTerm = incompleteGammaComputer<Real, rPower>::compute(rSquared * kappaSquared) /
                         gammaComputer<Real, rPower>::value;
        Real eKernel = gammaTerm / denominator;
        Real fKernel = -rPower * eKernel * rInv -
                       2 * rInv * exp(-kappaSquared * rSquared) * kappaToRPower / gammaComputer<Real, rPower>::value;
        return std::make_tuple(eKernel, fKernel);
    }

    /*!
     * \brief adjEImpl computes the kernel for the adjusted energy for a pair.
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param rSquared the square of the internuclear distance
     * \param kappaSquared the square of attenuation parameter in units inverse of those used to specify coordinates.
     * \return the energy kernel.
     */
    template <int rPower>
    inline static Real adjEImpl(Real rSquared, Real kappaSquared) {
        Real denominator = raiseNormToIntegerPower<Real, rPower>::compute(rSquared);
        Real gammaTerm = incompleteGammaComputer<Real, rPower>::compute(rSquared * kappaSquared) /
                         gammaComputer<Real, rPower>::value;
        return (gammaTerm - 1) / denominator;
    }

    /*!
     * \brief adjEFImpl computes the kernels for the adjusted energy and force for a pair.
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param rSquared the square of the internuclear distance
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param kappaSquared the square of attenuation parameter in units inverse of those used to specify coordinates.
     * \return a tuple containing the energy and force kernels, respectively.
     */
    template <int rPower>
    inline static std::tuple<Real, Real> adjEFImpl(Real rSquared, Real kappa, Real kappaSquared) {
        Real rInv = 1 / rSquared;
        Real kappaToRPower = kappa;
        for (int i = 1; i < rPower; ++i) kappaToRPower *= kappa;
        Real denominator = raiseNormToIntegerPower<Real, rPower>::compute(rSquared);
        Real gammaTerm = incompleteGammaComputer<Real, rPower>::compute(rSquared * kappaSquared) /
                         gammaComputer<Real, rPower>::value;
        Real eKernel = (gammaTerm - 1) / denominator;
        Real fKernel = -rPower * eKernel * rInv -
                       2 * rInv * exp(-kappaSquared * rSquared) * kappaToRPower / gammaComputer<Real, rPower>::value;
        return std::make_tuple(eKernel, fKernel);
    }

    /*!
     * \brief slfEImpl computes the self energy due to particles feeling their own potential.
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the 1 / [4
     * pi epslion0] for Coulomb calculations).
     * \return the self energy.  N.B. there is no self force associated with this term.
     */
    template <int rPower>
    static Real slfEImpl(int parameterAngMom, const RealMat &parameters, Real kappa, Real scaleFactor) {
        if (parameterAngMom) throw std::runtime_error("Multipole self terms have not been coded yet.");

        size_t nAtoms = parameters.nRows();
        Real prefac = -scaleFactor * std::pow(kappa, rPower) / (rPower * gammaComputer<Real, rPower>::value);
        Real sumCoefs = 0;
        for (size_t atom = 0; atom < nAtoms; ++atom) {
            sumCoefs += parameters(atom, 0) * parameters(atom, 0);
        }
        return prefac * sumCoefs;
    }

    /*!
     * \brief common_init sets up information that is common to serial and parallel runs.
     */
    void common_init(int rPower, Real kappa, int splineOrder, int dimA, int dimB, int dimC, Real scaleFactor,
                     int nThreads) {
        dimA_ = dimA;
        dimB_ = dimB;
        dimC_ = dimC;
        complexDimA_ = dimA / 2 + 1;
        myComplexDimA_ = myDimA_ / 2 + 1;
        splineOrder_ = splineOrder;
        nThreads_ = nThreads;
        scaleFactor_ = scaleFactor;
        kappa_ = kappa;

        // Helpers to perform 1D FFTs along each dimension.
        fftHelperA_ = FFTWWrapper<Real>(dimA_);
        fftHelperB_ = FFTWWrapper<Real>(dimB_);
        fftHelperC_ = FFTWWrapper<Real>(dimC_);

        // Grid iterators to correctly wrap the grid when using splines.
        gridIteratorA_ = makeGridIterator(dimA_, firstA_, lastA_);
        gridIteratorB_ = makeGridIterator(dimB_, firstB_, lastB_);
        gridIteratorC_ = makeGridIterator(dimC_, firstC_, lastC_);

        // Fourier space spline norms.
        BSpline<Real> spline = BSpline<Real>(0, 0, splineOrder_, 0);
        splineModA_ = spline.invSplineModuli(dimA_);
        splineModB_ = spline.invSplineModuli(dimB_);
        splineModC_ = spline.invSplineModuli(dimC_);

        // Set up function pointers by instantiating the appropriate evaluation functions.  We could add many more
        // entries by default here, but don't right now to avoid code bloat.  To add an extra rPower kernel is a
        // trivial cut and paste exercise; just add a new line with the desired 1/R power as the macro's argument.
        switch (rPower) {
            ENABLE_KERNEL_WITH_INVERSE_R_EXPONENT_OF(1);
            ENABLE_KERNEL_WITH_INVERSE_R_EXPONENT_OF(6);
            default:
                std::string msg("Bad rPower requested.  To fix this, add the appropriate entry in");
                msg += __FILE__;
                msg += ", line number ";
                msg += std::to_string(__LINE__ - 5);
                throw std::runtime_error(msg.c_str());
                break;
        }

        subsetOfCAlongA_ = myDimC_ / numNodesA_;
        subsetOfCAlongB_ = myDimC_ / numNodesB_;
        subsetOfCAlongC_ = myDimC_ / numNodesC_;
        subsetOfBAlongC_ = myDimB_ / numNodesC_;

        workSpace1_ = helpme::vector<Complex>(myDimC_ * myComplexDimA_ * myDimB_);
        workSpace2_ = helpme::vector<Complex>(myDimC_ * myComplexDimA_ * myDimB_);
    }

   public:
    /*!
     * \brief The different conventions for orienting a lattice constructed from input parameters.
     */
    enum class LatticeType : int { XAligned = 0, ShapeMatrix = 1 };

    /*!
     * \brief The different conventions for numbering nodes.
     */
    enum class NodeOrder : int { ZYX = 0 };

    PMEInstance() : boxVecs_(3, 3), recVecs_(3, 3), scaledRecVecs_(3, 3) {}

    /*!
     * \brief cellVolume Compute the volume of the unit cell.
     * \return volume in units consistent with those used to define the lattice vectors.
     */
    Real cellVolume() {
        return boxVecs_(0, 0) * boxVecs_(1, 1) * boxVecs_(2, 2) - boxVecs_(0, 0) * boxVecs_(1, 2) * boxVecs_(2, 1) +
               boxVecs_(0, 1) * boxVecs_(1, 2) * boxVecs_(2, 0) - boxVecs_(0, 1) * boxVecs_(1, 0) * boxVecs_(2, 2) +
               boxVecs_(0, 2) * boxVecs_(1, 0) * boxVecs_(2, 1) - boxVecs_(0, 2) * boxVecs_(1, 1) * boxVecs_(2, 0);
    }

    /*!
     * \brief Sets the unit cell lattice vectors, with units consistent with those used to specify coordinates.
     * \param A the A lattice parameter in units consistent with the coordinates.
     * \param B the B lattice parameter in units consistent with the coordinates.
     * \param C the C lattice parameter in units consistent with the coordinates.
     * \param alpha the alpha lattice parameter in degrees.
     * \param beta the beta lattice parameter in degrees.
     * \param gamma the gamma lattice parameter in degrees.
     * \param latticeType how to arrange the lattice vectors.  Options are
     * ShapeMatrix: enforce a symmetric representation of the lattice vectors [c.f. S. Nosé and M. L. Klein,
     *              Mol. Phys. 50 1055 (1983)] particularly appendix C.
     * XAligned: make the A vector coincide with the X axis, the B vector fall in the XY plane, and the C vector
     *           take the appropriate alignment to completely define the system.
     */
    void setLatticeVectors(Real A, Real B, Real C, Real alpha, Real beta, Real gamma, LatticeType latticeType) {
        if (latticeType == LatticeType::ShapeMatrix) {
            RealMat HtH(3, 3);
            HtH(0, 0) = A * A;
            HtH(1, 1) = B * B;
            HtH(2, 2) = C * C;
            const float TOL = 1e-4f;
            // Check for angles very close to 90, to avoid noise from the eigensolver later on.
            HtH(0, 1) = HtH(1, 0) = std::abs(gamma - 90) < TOL ? 0 : A * B * cos(M_PI * gamma / 180);
            HtH(0, 2) = HtH(2, 0) = std::abs(beta - 90) < TOL ? 0 : A * C * cos(M_PI * beta / 180);
            HtH(1, 2) = HtH(2, 1) = std::abs(alpha - 90) < TOL ? 0 : B * C * cos(M_PI * alpha / 180);

            auto eigenTuple = HtH.diagonalize();
            RealMat evalsReal = std::get<0>(eigenTuple);
            RealMat evalsImag = std::get<1>(eigenTuple);
            RealMat evecs = std::get<2>(eigenTuple);
            if (!evalsImag.isNearZero())
                throw std::runtime_error("Unexpected complex eigenvalues encountered while making shape matrix.");
            for (int i = 0; i < 3; ++i) evalsReal(i, 0) = sqrt(evalsReal(i, 0));
            boxVecs_.setZero();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        boxVecs_(i, j) += evecs(i, k) * evecs(j, k) * evalsReal(k, 0);
                    }
                }
            }
            recVecs_ = boxVecs_.inverse();
        } else if (latticeType == LatticeType::XAligned) {
            boxVecs_(0, 0) = A;
            boxVecs_(0, 1) = 0;
            boxVecs_(0, 2) = 0;
            boxVecs_(1, 0) = B * cos(M_PI / 180 * gamma);
            boxVecs_(1, 1) = B * sin(M_PI / 180 * gamma);
            boxVecs_(1, 2) = 0;
            boxVecs_(2, 0) = C * cos(M_PI / 180 * beta);
            boxVecs_(2, 1) = (B * C * cos(M_PI / 180 * alpha) - boxVecs_(2, 0) * boxVecs_(1, 0)) / boxVecs_(1, 1);
            boxVecs_(2, 2) = sqrt(C * C - boxVecs_(2, 0) * boxVecs_(2, 0) - boxVecs_(2, 1) * boxVecs_(2, 1));
        } else {
            throw std::runtime_error("Unknown lattice type in setLatticeVectors");
        }
        recVecs_ = boxVecs_.inverse();
        scaledRecVecs_ = recVecs_.clone();
        scaledRecVecs_.row(0) *= dimA_;
        scaledRecVecs_.row(1) *= dimB_;
        scaledRecVecs_.row(2) *= dimC_;
    }

    /*!
     * \brief Performs the forward 3D FFT of the discretized parameter grid.
     * \param realGrid the array of discretized parameters (stored in CBA order,
     *                 with A being the fast running index) to be transformed.
     * \return Pointer to the transformed grid, which is stored in one of the buffers in BAC order.
     */
    Complex *forwardTransform(Real *realGrid) {
        Real *realCBA;
        Complex *buffer1, *buffer2;
        if (realGrid == reinterpret_cast<Real *>(workSpace1_.data())) {
            realCBA = reinterpret_cast<Real *>(workSpace2_.data());
            buffer1 = workSpace2_.data();
            buffer2 = workSpace1_.data();
        } else {
            realCBA = reinterpret_cast<Real *>(workSpace2_.data());
            buffer1 = workSpace2_.data();
            buffer2 = workSpace1_.data();
        }

#if HAVE_MPI == 1
        if (numNodesA_ > 1) {
            // Communicate A along columns
            mpiCommunicatorA_->allToAll(realGrid, realCBA, subsetOfCAlongA_ * myDimA_ * myDimB_);
            // Resort the data to end up with realGrid holding a full row of A data, for B pencil and C subset.
            for (int c = 0; c < subsetOfCAlongA_; ++c) {
                Real *outC = realGrid + c * myDimB_ * dimA_;
                for (int b = 0; b < myDimB_; ++b) {
                    for (int chunk = 0; chunk < numNodesA_; ++chunk) {
                        Real *inPtr = realCBA + (chunk * subsetOfCAlongA_ + c) * myDimB_ * myDimA_ + b * myDimA_;
                        std::copy(inPtr, inPtr + myDimA_, outC + b * dimA_ + chunk * myDimA_);
                    }
                }
            }
        }
#endif

        // A transform, with instant sort to CAB ordering for each local block
        helpme::vector<Complex> buffer(complexDimA_);
        auto scratch = buffer.data();
        for (int c = 0; c < subsetOfCAlongA_; ++c) {
            for (int b = 0; b < myDimB_; ++b) {
                Real *gridPtr = realGrid + c * myDimB_ * dimA_ + b * dimA_;
                fftHelperA_.transform(gridPtr, scratch);
                for (int chunk = 0; chunk < numNodesA_; ++chunk) {
                    // The overall transform dim is numA/2+1.  We stuff the extra element in the 0th block.
                    int aShift = (chunk ? 1 : 0);
                    for (int a = 0; a < myComplexDimA_ - aShift; ++a) {
                        buffer1[(chunk * subsetOfCAlongA_ + c) * myComplexDimA_ * myDimB_ + a * myDimB_ + b] =
                            scratch[chunk * (myComplexDimA_ - 1) + a + aShift];
                    }
                }
            }
        }

#if HAVE_MPI == 1
        // Communicate A back to blocks
        if (numNodesA_ > 1) {
            mpiCommunicatorA_->allToAll(buffer1, buffer2, subsetOfCAlongA_ * myComplexDimA_ * myDimB_);
            std::swap(buffer1, buffer2);
        }

        // Communicate B along rows
        if (numNodesB_ > 1) {
            mpiCommunicatorB_->allToAll(buffer1, buffer2, subsetOfCAlongB_ * myComplexDimA_ * myDimB_);
            // Resort the data to end up with the buffer holding a full row of B data, for A pencil and C subset.
            for (int c = 0; c < subsetOfCAlongB_; ++c) {
                Complex *cPtr = buffer1 + c * myComplexDimA_ * dimB_;
                for (int a = 0; a < myComplexDimA_; ++a) {
                    for (int chunk = 0; chunk < numNodesB_; ++chunk) {
                        Complex *inPtr =
                            buffer2 + (chunk * subsetOfCAlongB_ + c) * myComplexDimA_ * myDimB_ + a * myDimB_;
                        std::copy(inPtr, inPtr + myDimB_, cPtr + a * dimB_ + chunk * myDimB_);
                    }
                }
            }
        }
#endif

        // B transform
        for (int c = 0; c < subsetOfCAlongB_; ++c) {
            Complex *cPtr = buffer1 + c * myComplexDimA_ * dimB_;
            for (int a = 0; a < myComplexDimA_; ++a) {
                fftHelperB_.transform(cPtr + a * dimB_, FFTW_FORWARD);
            }
        }

#if HAVE_MPI == 1
        if (numNodesB_ > 1) {
            for (int c = 0; c < subsetOfCAlongB_; ++c) {
                Complex *zPtr = buffer1 + c * myComplexDimA_ * dimB_;
                for (int a = 0; a < myComplexDimA_; ++a) {
                    for (int chunk = 0; chunk < numNodesB_; ++chunk) {
                        Complex *inPtr = zPtr + a * dimB_ + chunk * myDimB_;
                        Complex *outPtr =
                            buffer2 + (chunk * subsetOfCAlongB_ + c) * myComplexDimA_ * myDimB_ + a * myDimB_;
                        std::copy(inPtr, inPtr + myDimB_, outPtr);
                    }
                }
            }
            // Communicate B back to blocks
            mpiCommunicatorB_->allToAll(buffer2, buffer1, subsetOfCAlongB_ * myComplexDimA_ * myDimB_);
        }
#endif

        // sort local blocks from CAB to BAC order
        for (int b = 0; b < myDimB_; ++b) {
            for (int a = 0; a < myComplexDimA_; ++a) {
                for (int c = 0; c < myDimC_; ++c) {
                    buffer2[b * myComplexDimA_ * myDimC_ + a * myDimC_ + c] =
                        buffer1[c * myComplexDimA_ * myDimB_ + a * myDimB_ + b];
                }
            }
        }

#if HAVE_MPI == 1
        if (numNodesC_ > 1) {
            // Communicate C along columns
            mpiCommunicatorC_->allToAll(buffer2, buffer1, subsetOfCAlongC_ * myComplexDimA_ * myDimB_);
            for (int b = 0; b < subsetOfBAlongC_; ++b) {
                Complex *outPtrB = buffer2 + b * myComplexDimA_ * dimC_;
                for (int a = 0; a < myComplexDimA_; ++a) {
                    Complex *outPtrBA = outPtrB + a * dimC_;
                    for (int chunk = 0; chunk < numNodesC_; ++chunk) {
                        Complex *inPtr =
                            buffer1 + (chunk * subsetOfBAlongC_ + b) * myComplexDimA_ * myDimC_ + a * myDimC_;
                        std::copy(inPtr, inPtr + myDimC_, outPtrBA + chunk * myDimC_);
                    }
                }
            }
        }
#endif

        // C transform
        for (int b = 0; b < subsetOfBAlongC_; ++b) {
            Complex *outPtrB = buffer2 + b * myComplexDimA_ * dimC_;
            for (int a = 0; a < myComplexDimA_; ++a) {
                Complex *outPtrBA = outPtrB + a * dimC_;
                fftHelperC_.transform(outPtrBA, FFTW_FORWARD);
            }
        }

        return buffer2;
    }

    /*!
     * \brief Performs the inverse 3D FFT.
     * \param convolvedGrid the complex array of discretized parameters convolved with the influence function
     *                      (stored in BAC order, with C being the fast running index) to be transformed.
     * \return Pointer to the potential grid, which is stored in one of the buffers in CBA order.
     */
    Real *inverseTransform(Complex *convolvedGrid) {
        Complex *buffer1, *buffer2;
        // Setup scratch, taking care not to overwrite the convolved grid.
        if (convolvedGrid == workSpace1_.data()) {
            buffer1 = workSpace2_.data();
            buffer2 = workSpace1_.data();
        } else {
            buffer1 = workSpace1_.data();
            buffer2 = workSpace2_.data();
        }

        // C transform
        for (int y = 0; y < subsetOfBAlongC_; ++y) {
            for (int x = 0; x < myComplexDimA_; ++x) {
                int yx = y * myComplexDimA_ * dimC_ + x * dimC_;
                fftHelperC_.transform(convolvedGrid + yx, FFTW_BACKWARD);
            }
        }

#if HAVE_MPI == 1
        if (numNodesC_ > 1) {
            // Communicate C back to blocks
            for (int b = 0; b < subsetOfBAlongC_; ++b) {
                Complex *inPtrB = convolvedGrid + b * myComplexDimA_ * dimC_;
                for (int a = 0; a < myComplexDimA_; ++a) {
                    Complex *inPtrBA = inPtrB + a * dimC_;
                    for (int chunk = 0; chunk < numNodesC_; ++chunk) {
                        Complex *inPtrBAC = inPtrBA + chunk * myDimC_;
                        Complex *outPtr =
                            buffer1 + (chunk * subsetOfBAlongC_ + b) * myComplexDimA_ * myDimC_ + a * myDimC_;
                        std::copy(inPtrBAC, inPtrBAC + myDimC_, outPtr);
                    }
                }
            }
            mpiCommunicatorC_->allToAll(buffer1, buffer2, subsetOfCAlongC_ * myComplexDimA_ * myDimB_);
        }
#endif

        // sort local blocks from BAC to CAB order
        for (int B = 0; B < myDimB_; ++B) {
            for (int A = 0; A < myComplexDimA_; ++A) {
                for (int C = 0; C < myDimC_; ++C) {
                    buffer1[C * myComplexDimA_ * myDimB_ + A * myDimB_ + B] =
                        buffer2[B * myComplexDimA_ * myDimC_ + A * myDimC_ + C];
                }
            }
        }

#if HAVE_MPI == 1
        // Communicate B along rows
        if (numNodesB_ > 1) {
            mpiCommunicatorB_->allToAll(buffer1, buffer2, subsetOfCAlongB_ * myComplexDimA_ * myDimB_);
            // Resort the data to end up with the buffer holding a full row of B data, for A pencil and C subset.
            for (int c = 0; c < subsetOfCAlongB_; ++c) {
                Complex *cPtr = buffer1 + c * myComplexDimA_ * dimB_;
                for (int a = 0; a < myComplexDimA_; ++a) {
                    for (int chunk = 0; chunk < numNodesB_; ++chunk) {
                        Complex *inPtr =
                            buffer2 + (chunk * subsetOfCAlongB_ + c) * myComplexDimA_ * myDimB_ + a * myDimB_;
                        std::copy(inPtr, inPtr + myDimB_, cPtr + a * dimB_ + chunk * myDimB_);
                    }
                }
            }
        }
#endif

        // B transform with instant sort of local blocks from CAB -> CBA order
        for (int c = 0; c < subsetOfCAlongB_; ++c) {
            for (int a = 0; a < myComplexDimA_; ++a) {
                int cx = c * myComplexDimA_ * dimB_ + a * dimB_;
                fftHelperB_.transform(buffer1 + cx, FFTW_BACKWARD);
                for (int b = 0; b < myDimB_; ++b) {
                    for (int chunk = 0; chunk < numNodesB_; ++chunk) {
                        int cb = (chunk * subsetOfCAlongB_ + c) * myDimB_ * myComplexDimA_ + b * myComplexDimA_;
                        buffer2[cb + a] = buffer1[cx + chunk * myDimB_ + b];
                    }
                }
            }
        }

#if HAVE_MPI == 1
        // Communicate B back to blocks
        if (numNodesB_ > 1) {
            mpiCommunicatorB_->allToAll(buffer2, buffer1, subsetOfCAlongB_ * myComplexDimA_ * myDimB_);
        } else {
            std::swap(buffer1, buffer2);
        }

        // Communicate A along rows
        if (numNodesA_ > 1) {
            mpiCommunicatorA_->allToAll(buffer1, buffer2, subsetOfCAlongA_ * myComplexDimA_ * myDimB_);
            // Resort the data to end up with the buffer holding a full row of A data, for B pencil and C subset.
            for (int c = 0; c < subsetOfCAlongA_; ++c) {
                Complex *cPtr = buffer1 + c * myDimB_ * complexDimA_;
                for (int b = 0; b < myDimB_; ++b) {
                    for (int chunk = 0; chunk < numNodesA_; ++chunk) {
                        int aShift = (chunk ? 1 : 0);
                        Complex *inPtr =
                            buffer2 + (chunk * subsetOfCAlongA_ + c) * myComplexDimA_ * myDimB_ + b * myComplexDimA_;
                        std::copy(inPtr, inPtr + myComplexDimA_ - aShift,
                                  cPtr + b * complexDimA_ + chunk * (myComplexDimA_ - 1) + aShift);
                    }
                }
            }
        }
#else
        std::swap(buffer1, buffer2);
#endif

        // A transform
        Real *realGrid = reinterpret_cast<Real *>(buffer2);
        for (int cb = 0; cb < subsetOfCAlongA_ * myDimB_; ++cb) {
            fftHelperA_.transform(buffer1 + cb * complexDimA_, realGrid + cb * dimA_);
        }

#if HAVE_MPI == 1
        // Communicate A back to blocks
        if (numNodesA_ > 1) {
            Real *realGrid2 = reinterpret_cast<Real *>(buffer1);
            for (int c = 0; c < subsetOfCAlongA_; ++c) {
                Real *cPtr = realGrid + c * myDimB_ * dimA_;
                for (int b = 0; b < myDimB_; ++b) {
                    for (int chunk = 0; chunk < numNodesA_; ++chunk) {
                        Real *outPtr = realGrid2 + (chunk * subsetOfCAlongA_ + c) * myDimB_ * myDimA_ + b * myDimA_;
                        Real *inPtr = cPtr + b * dimA_ + chunk * myDimA_;
                        std::copy(inPtr, inPtr + myDimA_, outPtr);
                    }
                }
            }
            mpiCommunicatorA_->allToAll(realGrid2, realGrid, subsetOfCAlongA_ * myDimB_ * myDimA_);
        }
#endif
        return realGrid;
    }

    /*!
     * \brief convolveE A wrapper to determine the correct convolution function to call.
     * \param transformedGrid the pointer to the complex array holding the transformed grid in YXZ ordering.
     * \return the reciprocal space energy.
     */
    Real convolveE(Complex *transformedGrid) {
        return convolveEFxn_(dimA_, dimB_, dimC_, myComplexDimA_, myDimB_ / numNodesC_,
                             rankA_ * myComplexDimA_ - (rankA_ ? rankA_ - 1 : 0),
                             rankB_ * myDimB_ + rankC_ * myDimB_ / numNodesC_, scaleFactor_, transformedGrid, recVecs_,
                             cellVolume(), kappa_, splineModA_, splineModB_, splineModC_);
    }

    /*!
     * \brief convolveEV A wrapper to determine the correct convolution function to call, including virial.
     * \param transformedGrid the pointer to the complex array holding the transformed grid in YXZ ordering.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \return the reciprocal space energy.
     */
    Real convolveEV(Complex *transformedGrid, RealMat &virial) {
        return convolveEVFxn_(dimA_, dimB_, dimC_, myComplexDimA_, myDimB_ / numNodesC_,
                              rankA_ * myComplexDimA_ - (rankA_ ? rankA_ - 1 : 0),
                              rankB_ * myDimB_ + rankC_ * myDimB_ / numNodesC_, scaleFactor_, transformedGrid, recVecs_,
                              cellVolume(), kappa_, splineModA_, splineModB_, splineModC_, virial);
    }

    /*!
     * \brief Spread the parameters onto the charge grid.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * Generally this shouldn't be called; use runPME() instead.
     */
    Real *spreadParameters(int parameterAngMom, const RealMat &parameters, const RealMat &coordinates) {
        Real *realGrid = reinterpret_cast<Real *>(workSpace1_.data());
        std::fill(workSpace1_.begin(), workSpace1_.end(), 0);
        updateAngMomIterator(parameterAngMom);
        int nComponents = nCartesian(parameterAngMom);
        size_t nAtoms = coordinates.nRows();
        for (size_t atom = 0; atom < nAtoms; ++atom) {
            auto bSplines = makeBSplines(coordinates[atom], parameterAngMom);
            auto splineA = std::get<0>(bSplines);
            auto splineB = std::get<1>(bSplines);
            auto splineC = std::get<2>(bSplines);
            const auto &aGridIterator = gridIteratorA_[splineA.startingGridPoint()];
            const auto &bGridIterator = gridIteratorB_[splineB.startingGridPoint()];
            const auto &cGridIterator = gridIteratorC_[splineC.startingGridPoint()];
            for (int component = 0; component < nComponents; ++component) {
                const auto &quanta = angMomIterator_[component];
                Real param = parameters(atom, component);
                const Real *splineValsA = splineA[quanta[0]];
                const Real *splineValsB = splineB[quanta[1]];
                const Real *splineValsC = splineC[quanta[2]];
                for (const auto &cPoint : cGridIterator) {
                    Real cValP = param * splineValsC[cPoint.second];
                    for (const auto &bPoint : bGridIterator) {
                        Real cbValP = cValP * splineValsB[bPoint.second];
                        Real *cbRow = realGrid + cPoint.first * myDimB_ * myDimA_ + bPoint.first * myDimA_;
                        for (const auto &aPoint : aGridIterator) {
                            cbRow[aPoint.first] += cbValP * splineValsA[aPoint.second];
                        }
                    }
                }
            }
        }
        return realGrid;
    }

    /*!
     * \brief Probes the potential grid to get the forces.
     *
     * Generally this shouldn't be called; use runPME() instead.
     * \param potentialGrid pointer to the array containing the potential, in ZYX order.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     */
    void probeGrid(const Real *potentialGrid, int parameterAngMom, const RealMat &parameters,
                   const RealMat &coordinates, RealMat &forces) {
        updateAngMomIterator(parameterAngMom + 1);
        int nComponents = nCartesian(parameterAngMom);
        int nForceComponents = nCartesian(parameterAngMom + 1);
        size_t nAtoms = coordinates.nRows();
        for (size_t atom = 0; atom < nAtoms; ++atom) {
            RealMat fractionalPhis(1, nForceComponents);
            fractionalPhis.setZero();

            auto bSplines = makeBSplines(coordinates[atom], parameterAngMom + 1);
            auto splineA = std::get<0>(bSplines);
            auto splineB = std::get<1>(bSplines);
            auto splineC = std::get<2>(bSplines);
            const auto &aGridIterator = gridIteratorA_[splineA.startingGridPoint()];
            const auto &bGridIterator = gridIteratorB_[splineB.startingGridPoint()];
            const auto &cGridIterator = gridIteratorC_[splineC.startingGridPoint()];
            for (const auto &cPoint : cGridIterator) {
                for (const auto &bPoint : bGridIterator) {
                    const Real *cbRow = potentialGrid + cPoint.first * myDimA_ * myDimB_ + bPoint.first * myDimA_;
                    for (const auto &aPoint : aGridIterator) {
                        Real gridVal = cbRow[aPoint.first];
                        for (int component = 0; component < nForceComponents; ++component) {
                            const auto &quanta = angMomIterator_[component];
                            const Real *splineValsA = splineA[quanta[0]];
                            const Real *splineValsB = splineB[quanta[1]];
                            const Real *splineValsC = splineC[quanta[2]];
                            fractionalPhis[0][component] += gridVal * splineValsA[aPoint.second] *
                                                            splineValsB[bPoint.second] * splineValsC[cPoint.second];
                        }
                    }
                }
            }

            Real fracForce[3] = {0, 0, 0};
            for (int component = 0; component < nComponents; ++component) {
                Real param = parameters(atom, component);
                const auto &quanta = angMomIterator_[component];
                short lx = quanta[0];
                short ly = quanta[1];
                short lz = quanta[2];
                fracForce[0] += param * fractionalPhis(0, cartAddress(lx + 1, ly, lz));
                fracForce[1] += param * fractionalPhis(0, cartAddress(lx, ly + 1, lz));
                fracForce[2] += param * fractionalPhis(0, cartAddress(lx, ly, lz + 1));
            }
            forces(atom, 0) += scaledRecVecs_[0][0] * fracForce[0] + scaledRecVecs_[0][1] * fracForce[1] +
                               scaledRecVecs_[0][2] * fracForce[2];
            forces(atom, 1) += scaledRecVecs_[1][0] * fracForce[0] + scaledRecVecs_[1][1] * fracForce[1] +
                               scaledRecVecs_[1][2] * fracForce[2];
            forces(atom, 2) += scaledRecVecs_[2][0] * fracForce[0] + scaledRecVecs_[2][1] * fracForce[1] +
                               scaledRecVecs_[2][2] * fracForce[2];
        }
    }

    /*!
     * \brief computeESlf computes the Ewald self interaction energy.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \return the self energy.
     */
    Real computeESlf(int parameterAngMom, const RealMat &parameters) {
        return slfEFxn_(parameterAngMom, parameters, kappa_, scaleFactor_);
    }

    /*!
     * \brief computeEDir computes the direct space energy.  This is provided mostly for debugging and testing purposes;
     *        generally the host program should provide the pairwise interactions.
     * \param pairList dense list of atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \return the direct space energy.
     */
    Real computeEDir(const Matrix<short> &pairList, int parameterAngMom, const RealMat &parameters,
                     const RealMat &coordinates) {
        if (parameterAngMom) throw std::runtime_error("Multipole self terms have not been coded yet.");

        Real energy = 0;
        Real kappaSquared = kappa_ * kappa_;
        size_t nPair = pairList.nRows();
        for (int pair = 0; pair < nPair; ++pair) {
            short i = pairList(pair, 0);
            short j = pairList(pair, 1);
            auto deltaR = coordinates.row(j) - coordinates.row(i);
            // TODO: apply minimum image convention.
            Real rSquared = deltaR.dot(deltaR);
            energy += parameters(i, 0) * parameters(j, 0) * dirEFxn_(rSquared, kappaSquared);
        }
        return scaleFactor_ * energy;
    }

    /*!
     * \brief computeEFDir computes the direct space energy and force.  This is provided mostly for debugging and
     * testing purposes; generally the host program should provide the pairwise interactions.
     * \param pairList dense list of atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \return the direct space energy.
     */
    Real computeEFDir(const Matrix<short> &pairList, int parameterAngMom, const RealMat &parameters,
                      const RealMat &coordinates, RealMat &forces) {
        if (parameterAngMom) throw std::runtime_error("Multipole self terms have not been coded yet.");

        Real energy = 0;
        Real kappaSquared = kappa_ * kappa_;
        size_t nPair = pairList.nRows();
        for (int pair = 0; pair < nPair; ++pair) {
            short i = pairList(pair, 0);
            short j = pairList(pair, 1);
            auto deltaR = coordinates.row(j) - coordinates.row(i);
            // TODO: apply minimum image convention.
            Real rSquared = deltaR.dot(deltaR);
            auto kernels = dirEFFxn_(rSquared, kappa_, kappaSquared);
            Real eKernel = std::get<0>(kernels);
            Real fKernel = std::get<1>(kernels);
            Real prefactor = scaleFactor_ * parameters(i, 0) * parameters(j, 0);
            energy += prefactor * eKernel;
            Real f = prefactor * fKernel;
            auto force = deltaR.row(0);
            force *= f;
            forces.row(i) -= force;
            forces.row(j) += force;
        }
        return energy;
    }

    /*!
     * \brief computeEFVDir computes the direct space energy, force and virial.  This is provided mostly for debugging
     *        and testing purposes; generally the host program should provide the pairwise interactions.
     * \param pairList dense list of atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \return the direct space energy.
     */
    Real computeEFVDir(const Matrix<short> &pairList, int parameterAngMom, const RealMat &parameters,
                       const RealMat &coordinates, RealMat &forces, RealMat &virial) {
        if (parameterAngMom) throw std::runtime_error("Multipole self terms have not been coded yet.");

        Real energy = 0;
        Real kappaSquared = kappa_ * kappa_;
        size_t nPair = pairList.nRows();
        for (int pair = 0; pair < nPair; ++pair) {
            short i = pairList(pair, 0);
            short j = pairList(pair, 1);
            auto deltaR = coordinates.row(j) - coordinates.row(i);
            // TODO: apply minimum image convention.
            Real rSquared = deltaR.dot(deltaR);
            auto kernels = dirEFFxn_(rSquared, kappa_, kappaSquared);
            Real eKernel = std::get<0>(kernels);
            Real fKernel = std::get<1>(kernels);
            Real prefactor = scaleFactor_ * parameters(i, 0) * parameters(j, 0);
            energy += prefactor * eKernel;
            Real f = prefactor * fKernel;
            RealMat dRCopy = deltaR.clone();
            auto force = dRCopy.row(0);
            force *= f;
            forces.row(i) -= force;
            forces.row(j) += force;
            virial[0][0] -= force[0] * deltaR[0][0];
            virial[0][1] -= 0.5f * (force[0] * deltaR[0][1] + force[1] * deltaR[0][0]);
            virial[0][2] -= force[1] * deltaR[0][1];
            virial[0][3] -= 0.5f * (force[0] * deltaR[0][2] + force[2] * deltaR[0][0]);
            virial[0][4] -= 0.5f * (force[1] * deltaR[0][2] + force[2] * deltaR[0][1]);
            virial[0][5] -= force[2] * deltaR[0][2];
        }
        return energy;
    }

    /*!
     * \brief computeEAdj computes the adjusted real space energy which extracts the energy for excluded pairs that is
     *        present in reciprocal space. This is provided mostly for debugging and testing purposes; generally the
     *        host program should provide the pairwise interactions.
     * \param pairList dense list of atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \return the adjusted energy.
     */
    Real computeEAdj(const Matrix<short> &pairList, int parameterAngMom, const RealMat &parameters,
                     const RealMat &coordinates) {
        if (parameterAngMom) throw std::runtime_error("Multipole self terms have not been coded yet.");

        Real energy = 0;
        Real kappaSquared = kappa_ * kappa_;
        size_t nPair = pairList.nRows();
        for (int pair = 0; pair < nPair; ++pair) {
            short i = pairList(pair, 0);
            short j = pairList(pair, 1);
            auto deltaR = coordinates.row(j) - coordinates.row(i);
            // TODO: apply minimum image convention.
            Real rSquared = deltaR.dot(deltaR);
            energy += parameters(i, 0) * parameters(j, 0) * adjEFxn_(rSquared, kappaSquared);
        }
        return scaleFactor_ * energy;
    }

    /*!
     * \brief computeEFAdj computes the adjusted energy and force.  This is provided mostly for debugging and testing
     * purposes; generally the host program should provide the pairwise interactions.
     * \param pairList dense list of atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \return the adjusted energy.
     */
    Real computeEFAdj(const Matrix<short> &pairList, int parameterAngMom, const RealMat &parameters,
                      const RealMat &coordinates, RealMat &forces) {
        if (parameterAngMom) throw std::runtime_error("Multipole self terms have not been coded yet.");

        Real energy = 0;
        Real kappaSquared = kappa_ * kappa_;
        size_t nPair = pairList.nRows();
        for (int pair = 0; pair < nPair; ++pair) {
            short i = pairList(pair, 0);
            short j = pairList(pair, 1);
            auto deltaR = coordinates.row(j) - coordinates.row(i);
            // TODO: apply minimum image convention.
            Real rSquared = deltaR.dot(deltaR);
            auto kernels = adjEFFxn_(rSquared, kappa_, kappaSquared);
            Real eKernel = std::get<0>(kernels);
            Real fKernel = std::get<1>(kernels);
            Real prefactor = scaleFactor_ * parameters(i, 0) * parameters(j, 0);
            energy += prefactor * eKernel;
            Real f = prefactor * fKernel;
            auto force = deltaR.row(0);
            force *= f;
            forces.row(i) -= force;
            forces.row(j) += force;
        }
        return energy;
    }

    /*!
     * \brief computeEFVAdj computes the adjusted energy, forces and virial.  This is provided mostly for debugging
     *        and testing purposes; generally the host program should provide the pairwise interactions.
     * \param pairList dense list of atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \return the adjusted energy.
     */
    Real computeEFVAdj(const Matrix<short> &pairList, int parameterAngMom, const RealMat &parameters,
                       const RealMat &coordinates, RealMat &forces, RealMat &virial) {
        if (parameterAngMom) throw std::runtime_error("Multipole self terms have not been coded yet.");

        Real energy = 0;
        Real kappaSquared = kappa_ * kappa_;
        size_t nPair = pairList.nRows();
        for (int pair = 0; pair < nPair; ++pair) {
            short i = pairList(pair, 0);
            short j = pairList(pair, 1);
            auto deltaR = coordinates.row(j) - coordinates.row(i);
            // TODO: apply minimum image convention.
            Real rSquared = deltaR.dot(deltaR);
            auto kernels = adjEFFxn_(rSquared, kappa_, kappaSquared);
            Real eKernel = std::get<0>(kernels);
            Real fKernel = std::get<1>(kernels);
            Real prefactor = scaleFactor_ * parameters(i, 0) * parameters(j, 0);
            energy += prefactor * eKernel;
            Real f = prefactor * fKernel;
            RealMat dRCopy = deltaR.clone();
            auto force = dRCopy.row(0);
            force *= f;
            forces.row(i) -= force;
            forces.row(j) += force;
            virial[0][0] -= force[0] * deltaR[0][0];
            virial[0][1] -= 0.5f * (force[0] * deltaR[0][1] + force[1] * deltaR[0][0]);
            virial[0][2] -= force[1] * deltaR[0][1];
            virial[0][3] -= 0.5f * (force[0] * deltaR[0][2] + force[2] * deltaR[0][0]);
            virial[0][4] -= 0.5f * (force[1] * deltaR[0][2] + force[2] * deltaR[0][1]);
            virial[0][5] -= force[2] * deltaR[0][2];
        }
        return energy;
    }

    /*!
     * \brief Runs a PME reciprocal space calculation, computing energies.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param energy pointer to the variable holding the energy; this is incremented, not assigned.
     * \return the reciprocal space energy.
     */
    Real computeERec(int parameterAngMom, const RealMat &parameters, const RealMat &coordinates) {
        sanityChecks(parameterAngMom, parameters, coordinates);

        auto realGrid = spreadParameters(parameterAngMom, parameters, coordinates);
        auto gridAddress = forwardTransform(realGrid);
        return convolveE(gridAddress);
    }

    /*!
     * \brief Runs a PME reciprocal space calculation, computing energies and forces.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param energy pointer to the variable holding the energy; this is incremented, not assigned.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \return the reciprocal space energy.
     */
    Real computeEFRec(int parameterAngMom, const RealMat &parameters, const RealMat &coordinates, RealMat &forces) {
        sanityChecks(parameterAngMom, parameters, coordinates);

        auto realGrid = spreadParameters(parameterAngMom, parameters, coordinates);
        auto gridAddress = forwardTransform(realGrid);
        Real energy = convolveE(gridAddress);
        const auto potentialGrid = inverseTransform(gridAddress);
        probeGrid(potentialGrid, parameterAngMom, parameters, coordinates, forces);

        return energy;
    }

    /*!
     * \brief Runs a PME reciprocal space calculation, computing energies, forces and the virial.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param energy pointer to the variable holding the energy; this is incremented, not assigned.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \return the reciprocal space energy.
     */
    Real computeEFVRec(int parameterAngMom, const RealMat &parameters, const RealMat &coordinates, RealMat &forces,
                       RealMat &virial) {
        sanityChecks(parameterAngMom, parameters, coordinates);

        auto realGrid = spreadParameters(parameterAngMom, parameters, coordinates);
        auto gridPtr = forwardTransform(realGrid);
        Real energy = convolveEV(gridPtr, virial);
        const auto potentialGrid = inverseTransform(gridPtr);
        probeGrid(potentialGrid, parameterAngMom, parameters, coordinates, forces);

        return energy;
    }

    /*!
     * \brief Runs a full (direct and reciprocal space) PME calculation, computing the energy.  The direct space
     *        implementation here is not totally optimal, so this routine should primarily be used for testing and
     *        debugging.
     * \param includedList dense list of included atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN,jN.
     * \param excludedList dense list of excluded atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param energy pointer to the variable holding the energy; this is incremented, not assigned.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \return the full PME energy.
     */
    Real computeEAll(const Matrix<short> &includedList, const Matrix<short> &excludedList, int parameterAngMom,
                     const RealMat &parameters, const RealMat &coordinates) {
        Real energy = computeERec(parameterAngMom, parameters, coordinates);
        energy += computeESlf(parameterAngMom, parameters);
        energy += computeEDir(includedList, parameterAngMom, parameters, coordinates);
        energy += computeEAdj(excludedList, parameterAngMom, parameters, coordinates);
        return energy;
    }

    /*!
     * \brief Runs a full (direct and reciprocal space) PME calculation, computing energies and forces.  The direct
     *        space implementation here is not totally optimal, so this routine should primarily be used for testing
     *        and debugging.
     * \param includedList dense list of included atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param excludedList dense list of excluded atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param energy pointer to the variable holding the energy; this is incremented, not assigned.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \return the full PME energy.
     */
    Real computeEFAll(const Matrix<short> &includedList, const Matrix<short> &excludedList, int parameterAngMom,
                      const RealMat &parameters, const RealMat &coordinates, RealMat &forces) {
        Real energy = computeEFRec(parameterAngMom, parameters, coordinates, forces);
        energy += computeESlf(parameterAngMom, parameters);
        energy += computeEFDir(includedList, parameterAngMom, parameters, coordinates, forces);
        energy += computeEFAdj(excludedList, parameterAngMom, parameters, coordinates, forces);
        return energy;
    }

    /*!
     * \brief Runs a full (direct and reciprocal space) PME calculation, computing energies, forces and virials.
     *        The direct space implementation here is not totally optimal, so this routine should primarily
     *        be used for testing and debugging.
     * \param includedList dense list of included atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param excludedList dense list of excluded atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles,
     * etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles,
     * etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
     * (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
     *
     * 0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
     *
     * i.e. generated by the python loops
     * \code{.py}
     * for L in range(maxAM+1):
     *     for Lz in range(0,L+1):
     *         for Ly in range(0, L - Lz + 1):
     *              Lx  = L - Ly - Lz
     * \endcode
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param energy pointer to the variable holding the energy; this is incremented, not assigned.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \return the full PME energy.
     */
    Real computeEFVAll(const Matrix<short> &includedList, const Matrix<short> &excludedList, int parameterAngMom,
                       const RealMat &parameters, const RealMat &coordinates, RealMat &forces, RealMat &virial) {
        Real energy = computeEFVRec(parameterAngMom, parameters, coordinates, forces, virial);
        energy += computeESlf(parameterAngMom, parameters);
        energy += computeEFVDir(includedList, parameterAngMom, parameters, coordinates, forces, virial);
        energy += computeEFVAdj(excludedList, parameterAngMom, parameters, coordinates, forces, virial);
        return energy;
    }

    /*!
     * \brief makeGridIterator makes an iterator over the spline values that contribute to this node's grid
     *        in a given Cartesian dimension.  The iterator is of the form (grid point, spline index) and is
     *        sorted by increasing grid point, for cache efficiency.
     * \param dimension the dimension of the grid in the Cartesian dimension of interest.
     * \param first the first grid point in the Cartesian dimension to be handled by this node.
     * \param last the element past the last grid point in the Cartesian dimension to be handled by this node.
     * \return the vector of spline iterators for each starting grid point.
     */
    GridIterator makeGridIterator(int dimension, int first, int last) {
        // TODO make me private!
        GridIterator gridIterator;
        for (int gridStart = 0; gridStart < dimension; ++gridStart) {
            std::vector<std::pair<short, short>> splineIterator(splineOrder_);
            splineIterator.clear();
            for (int splineIndex = 0; splineIndex < splineOrder_; ++splineIndex) {
                int gridPoint = (splineIndex + gridStart) % dimension;
                if (gridPoint >= first && gridPoint < last)
                    splineIterator.push_back(std::make_pair(gridPoint - first, splineIndex));
            }
            std::sort(splineIterator.begin(), splineIterator.end());
            gridIterator.push_back(splineIterator);
        }
        return gridIterator;
    }

    /*!
     * \brief setup initializes this object for a PME calculation using only threading.
     * \param rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param splineOrder the order of B-spline; must be at least (2 + max. multipole order + deriv. level needed).
     * \param dimA the dimension of the FFT grid along the A axis.
     * \param dimB the dimension of the FFT grid along the B axis.
     * \param dimC the dimension of the FFT grid along the C axis.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param nThreads the maximum number of threads to use for each MPI instance; if set to 0 all available threads are
     *        used.
     */
    void setup(int rPower, Real kappa, int splineOrder, int dimA, int dimB, int dimC, Real scaleFactor, int nThreads) {
        numNodesA_ = numNodesB_ = numNodesC_ = 1;
        rankA_ = rankB_ = rankC_ = 0;
        firstA_ = firstB_ = firstC_ = 0;
        dimA = findGridSize(dimA, {1});
        dimB = findGridSize(dimB, {1});
        dimC = findGridSize(dimC, {1});
        lastA_ = dimA;
        lastB_ = dimB;
        lastC_ = dimC;
        myDimA_ = dimA;
        myDimB_ = dimB;
        myDimC_ = dimC;
        common_init(rPower, kappa, splineOrder, dimA, dimB, dimC, scaleFactor, nThreads);
    }

    /*!
     * \brief setup initializes this object for a PME calculation using MPI parallism and threading.
     * \param rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param splineOrder the order of B-spline; must be at least (2 + max. multipole order + deriv. level needed).
     * \param dimA the dimension of the FFT grid along the A axis.
     * \param dimB the dimension of the FFT grid along the B axis.
     * \param dimC the dimension of the FFT grid along the C axis.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param nThreads the maximum number of threads to use for each MPI instance; if set to 0 all available threads are
     * \param communicator the MPI communicator for the reciprocal space calcultion, which should already be
     *        initialized.
     * \param numNodesA the number of nodes to be used for the A dimension.
     * \param numNodesB the number of nodes to be used for the B dimension.
     * \param numNodesC the number of nodes to be used for the C dimension.
     */
    void setupParallel(int rPower, Real kappa, int splineOrder, int dimA, int dimB, int dimC, Real scaleFactor,
                       int nThreads, const MPI_Comm &communicator, NodeOrder nodeOrder, int numNodesA, int numNodesB,
                       int numNodesC) {
#if HAVE_MPI == 1
        mpiCommunicator_ =
            std::unique_ptr<MPIWrapper<Real>>(new MPIWrapper<Real>(communicator, numNodesA, numNodesB, numNodesC));
        switch (nodeOrder) {
            case (NodeOrder::ZYX):
                rankA_ = mpiCommunicator_->myRank_ % numNodesA;
                rankB_ = (mpiCommunicator_->myRank_ % (numNodesB * numNodesA)) / numNodesA;
                rankC_ = mpiCommunicator_->myRank_ / (numNodesB * numNodesA);
                mpiCommunicatorA_ = mpiCommunicator_->split(rankC_ * numNodesB + rankB_, rankA_);
                mpiCommunicatorB_ = mpiCommunicator_->split(rankC_ * numNodesA + rankA_, rankB_);
                mpiCommunicatorC_ = mpiCommunicator_->split(rankB_ * numNodesA + rankA_, rankC_);
                break;
            default:
                throw std::runtime_error("Unknown NodeOrder in setupParallel.");
        }
        numNodesA_ = numNodesA;
        numNodesB_ = numNodesB;
        numNodesC_ = numNodesC;
        dimA = findGridSize(dimA, {numNodesA});
        dimB = findGridSize(dimB, {numNodesB * numNodesC});
        dimC = findGridSize(dimC, {numNodesA * numNodesC, numNodesB * numNodesC});
        myDimA_ = dimA / numNodesA;
        myDimB_ = dimB / numNodesB;
        myDimC_ = dimC / numNodesC;
        firstA_ = rankA_ * myDimA_;
        firstB_ = rankB_ * myDimB_;
        firstC_ = rankC_ * myDimC_;
        lastA_ = rankA_ == numNodesA ? dimA : (rankA_ + 1) * myDimA_;
        lastB_ = rankB_ == numNodesB ? dimB : (rankB_ + 1) * myDimB_;
        lastC_ = rankC_ == numNodesC ? dimC : (rankC_ + 1) * myDimC_;

        common_init(rPower, kappa, splineOrder, dimA, dimB, dimC, scaleFactor, nThreads);

#else   // Have MPI
        throw std::runtime_error(
            "setupParallel called, but helpme was not compiled with MPI.  Make sure you compile with -DHAVE_MPI=1 in "
            "the list of compiler definitions.");
#endif  // Have MPI
    }
};
}  // Namespace helpme

using PMEInstanceD = helpme::PMEInstance<double>;
using PMEInstanceF = helpme::PMEInstance<float>;

#else

// C header
#include <stddef.h>
#if HAVE_MPI == 1
#include <mpi.h>
#endif

typedef enum { XAligned = 0, ShapeMatrix = 1 } LatticeType;
typedef enum { ZYX = 0 } NodeOrder;

typedef struct PMEInstance PMEInstance;
extern struct PMEInstance *helpme_createD();
extern struct PMEInstance *helpme_createF();
extern void helpme_destroyD(struct PMEInstance *pme);
extern void helpme_destroyF(struct PMEInstance *pme);
extern void helpme_setupD(struct PMEInstance *pme, int rPower, double kappa, int splineOrder, int aDim, int bDim,
                          int cDim, double scaleFactor, int nThreads);
extern void helpme_setupF(struct PMEInstance *pme, int rPower, float kappa, int splineOrder, int aDim, int bDim,
                          int cDim, float scaleFactor, int nThreads);
#if HAVE_MPI == 1
extern void helpme_setup_parallelD(PMEInstance *pme, int rPower, double kappa, int splineOrder, int dimA, int dimB,
                                   int dimC, double scaleFactor, int nThreads, MPI_Comm communicator,
                                   NodeOrder nodeOrder, int numNodesA, int numNodesB, int numNodesC);
extern void helpme_setup_parallelF(PMEInstance *pme, int rPower, float kappa, int splineOrder, int dimA, int dimB,
                                   int dimC, float scaleFactor, int nThreads, MPI_Comm communicator,
                                   NodeOrder nodeOrder, int numNodesA, int numNodesB, int numNodesC);
#endif  // HAVE_MPI
extern void helpme_set_lattice_vectorsD(struct PMEInstance *pme, double A, double B, double C, double kappa,
                                        double beta, double gamma, LatticeType latticeType);
extern void helpme_set_lattice_vectorsF(struct PMEInstance *pme, float A, float B, float C, float kappa, float beta,
                                        float gamma, LatticeType latticeType);
extern double helpme_compute_E_recD(struct PMEInstance *pme, size_t nAtoms, int parameterAngMom, double *parameters,
                                    double *coordinates);
extern float helpme_compute_E_recF(struct PMEInstance *pme, size_t nAtoms, int parameterAngMom, float *parameters,
                                   float *coordinates);
extern double helpme_compute_EF_recD(struct PMEInstance *pme, size_t nAtoms, int parameterAngMom, double *parameters,
                                     double *coordinates, double *forces);
extern float helpme_compute_EF_recF(struct PMEInstance *pme, size_t nAtoms, int parameterAngMom, float *parameters,
                                    float *coordinates, float *forces);
extern double helpme_compute_EFV_recD(struct PMEInstance *pme, size_t nAtoms, int parameterAngMom, double *parameters,
                                      double *coordinates, double *forces, double *virial);
extern float helpme_compute_EFV_recF(struct PMEInstance *pme, size_t nAtoms, int parameterAngMom, float *parameters,
                                     float *coordinates, float *forces, float *virial);
#endif  // C++/C
#endif  // Header guard
