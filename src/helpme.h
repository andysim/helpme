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
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "cartesiantransform.h"
#include "fftw_wrapper.h"
#include "gamma.h"
#include "gridsize.h"
#include "matrix.h"
#if HAVE_MKL == 1
#include "mkl.h"
#endif
#include "memory.h"
#if HAVE_MPI == 1
#include "mpi_wrapper.h"
#endif
#include "powers.h"
#include "splines.h"
#include "string_utils.h"
#include "tensor_utils.h"

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
#define ENABLE_KERNEL_WITH_INVERSE_R_EXPONENT_OF(n)                  \
    case n:                                                          \
        convolveEVFxn_ = &convolveEVImpl<n>;                         \
        convolveEVCompressedFxn_ = &convolveEVCompressedImpl<n>;     \
        cacheInfluenceFunctionFxn_ = &cacheInfluenceFunctionImpl<n>; \
        slfEFxn_ = &slfEImpl<n>;                                     \
        dirEFxn_ = &dirEImpl<n>;                                     \
        adjEFxn_ = &adjEImpl<n>;                                     \
        dirEFFxn_ = &dirEFImpl<n>;                                   \
        adjEFFxn_ = &adjEFImpl<n>;                                   \
        break;

/*!
 * \class splineCacheEntry
 * \brief A placeholder to encapsulate information about a given atom's splines
 */
template <typename Real>
struct SplineCacheEntry {
    BSpline<Real> aSpline, bSpline, cSpline;
    int absoluteAtomNumber;
    SplineCacheEntry(int order, int derivativeLevel)
        : aSpline(0, 0, order, derivativeLevel),
          bSpline(0, 0, order, derivativeLevel),
          cSpline(0, 0, order, derivativeLevel),
          absoluteAtomNumber(-1) {}
};

/*!
 * \class PMEInstance
 * \brief A class to encapsulate information related to a particle mesh Ewald calculation.
 *
 * By storing information related to a single PME calculation in this way, we allow multiple
 * instances to be created in calculations requiring multiple PMEs, e.g. for computing both
 * electrostatic and attractive dispersion terms using PME to handle long-range interactions.
 * \tparam Real the floating point type to use for arithmetic.
 */
template <typename Real, typename std::enable_if<std::is_floating_point<Real>::value, int>::type = 0>
class PMEInstance {
    using GridIterator = std::vector<std::vector<std::pair<short, short>>>;
    using Complex = std::complex<Real>;
    using Spline = BSpline<Real>;
    using RealMat = Matrix<Real>;
    using RealVec = helpme::vector<Real>;

   public:
    /*!
     * \brief The algorithm being used to solve for the reciprocal space quantities.
     */
    enum class AlgorithmType : int { Undefined = 0, PME = 1, CompressedPME = 2 };

    /*!
     * \brief The different conventions for orienting a lattice constructed from input parameters.
     */
    enum class LatticeType : int { Undefined = 0, XAligned = 1, ShapeMatrix = 2 };

    /*!
     * \brief The different conventions for numbering nodes.
     */
    enum class NodeOrder : int { Undefined = 0, ZYX = 1 };

    /*!
     * \brief The method used to converge induced dipoles
     */
    enum class PolarizationType : int { Mutual = 0, Direct = 1 };

   protected:
    /// The FFT grid dimensions in the {A,B,C} grid dimensions.
    int gridDimensionA_ = 0, gridDimensionB_ = 0, gridDimensionC_ = 0;
    /// The number of K vectors in the {A,B,C} dimensions.  Equal to dim{A,B,C} for PME, lower for cPME.
    int numKSumTermsA_ = 0, numKSumTermsB_ = 0, numKSumTermsC_ = 0;
    /// The number of K vectors in the {A,B,C} dimensions to be handled by this node in a parallel setup.
    int myNumKSumTermsA_ = 0, myNumKSumTermsB_ = 0, myNumKSumTermsC_ = 0;
    /// The full A dimension after real->complex transformation.
    int complexGridDimensionA_ = 0;
    /// The locally owned A dimension after real->complex transformation.
    int myComplexGridDimensionA_ = 0;
    /// The order of the cardinal B-Spline used for interpolation.
    int splineOrder_ = 0;
    /// The actual number of threads per MPI instance, and the number requested previously.
    int nThreads_ = -1, requestedNumberOfThreads_ = -1;
    /// The exponent of the (inverse) interatomic distance used in this kernel.
    int rPower_ = 0;
    /// The scale factor to apply to all energies and derivatives.
    Real scaleFactor_ = 0;
    /// The attenuation parameter, whose units should be the inverse of those used to specify coordinates.
    Real kappa_ = 0;
    /// The lattice vectors.
    RealMat boxVecs_ = RealMat(3, 3);
    /// The reciprocal lattice vectors.
    RealMat recVecs_ = RealMat(3, 3);
    /// The scaled reciprocal lattice vectors, for transforming forces from scaled fractional coordinates.
    RealMat scaledRecVecs_ = RealMat(3, 3);
    /// A list of the number of splines handle by each thread on this node.
    std::vector<size_t> numAtomsPerThread_;
    /// An iterator over angular momentum components.
    std::vector<std::array<short, 3>> angMomIterator_;
    /// From a given starting point on the {A,B,C} edge of the grid, lists all points to be handled, correctly wrapping
    /// around the end.
    GridIterator gridIteratorA_, gridIteratorB_, gridIteratorC_;
    /// The grid iterator for the C dimension, divided up by threads to avoid race conditions in parameter spreading.
    std::vector<GridIterator> threadedGridIteratorC_;
    /// The (inverse) bspline moduli to normalize the spreading / probing steps; these are folded into the convolution.
    RealVec splineModA_, splineModB_, splineModC_;
    /// The cached influence function involved in the convolution.
    RealVec cachedInfluenceFunction_;
    /// A function pointer to call the approprate function to implement convolution with virial for conventional PME,
    /// templated to the rPower value.
    std::function<Real(bool, int, int, int, int, int, int, int, Real, Complex *, const RealMat &, Real, Real,
                       const Real *, const Real *, const Real *, const int *, const int *, const int *, RealMat &, int)>
        convolveEVFxn_;
    /// A function pointer to call the approprate function to implement convolution with virial for comporessed PME,
    /// templated to the rPower value.
    std::function<Real(int, int, int, int, int, int, Real, const Real *, Real *, const RealMat &, Real, Real,
                       const Real *, const Real *, const Real *, const int *, const int *, const int *, RealMat &, int)>
        convolveEVCompressedFxn_;
    /// A function pointer to call the approprate function to implement cacheing of the influence function that appears
    //  in the convolution, templated to the rPower value.
    std::function<void(int, int, int, int, int, int, Real, RealVec &, const RealMat &, Real, Real, const Real *,
                       const Real *, const Real *, const int *, const int *, const int *, int)>
        cacheInfluenceFunctionFxn_;
    /// A function pointer to call the approprate function to compute self energy, templated to the rPower value.
    std::function<Real(int, Real, Real)> slfEFxn_;
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
    int numNodesA_ = 1, numNodesB_ = 1, numNodesC_ = 1;
    /// The rank of this node along the {A,B,C} dimensions.
    int myNodeRankA_ = 0, myNodeRankB_ = 0, myNodeRankC_ = 0;
    /// The first grid point that this node is responsible for in the {A,B,C} dimensions.
    int myFirstGridPointA_ = 0, myFirstGridPointB_ = 0, myFirstGridPointC_ = 0;
    /// The first K sum term that this node is responsible for.
    int firstKSumTermA_ = 0, firstKSumTermB_ = 0, firstKSumTermC_ = 0;
    /// The {X,Y,Z} dimensions of the locally owned chunk of the grid.
    int myGridDimensionA_ = 0, myGridDimensionB_ = 0, myGridDimensionC_ = 0;
    /// The subsets of a given dimension to be processed when doing a transform along another dimension.
    int subsetOfCAlongA_ = 0, subsetOfCAlongB_ = 0, subsetOfBAlongC_ = 0;
    /// The size of a cache line, in units of the size of the Real type, to allow better memory allocation policies.
    Real cacheLineSizeInReals_ = 0;
    /// The current unit cell parameters.
    Real cellA_ = 0, cellB_ = 0, cellC_ = 0, cellAlpha_ = 0, cellBeta_ = 0, cellGamma_ = 0;
    /// Whether the unit cell parameters have been changed, invalidating cached gF quantities.
    bool unitCellHasChanged_ = true;
    /// Whether the kappa has been changed, invalidating kappa-dependent quantities.
    bool kappaHasChanged_ = true;
    /// Whether any of the grid dimensions have changed.
    bool gridDimensionHasChanged_ = true;
    /// Whether any of the reciprocal sum dimensions have changed.
    bool reciprocalSumDimensionHasChanged_ = true;
    /// Whether the algorithm to be used has changed.
    bool algorithmHasChanged_ = true;
    /// Whether the spline order has changed.
    bool splineOrderHasChanged_ = true;
    /// Whether the scale factor has changed.
    bool scaleFactorHasChanged_ = true;
    /// Whether the power of R has changed.
    bool rPowerHasChanged_ = true;
    /// Whether the parallel node setup has changed in any way.
    bool numNodesHasChanged_ = true;
    /// The algorithm being used to solve for reciprocal space quantities.
    AlgorithmType algorithmType_ = AlgorithmType::Undefined;
    /// The type of alignment scheme used for the lattice vectors.
    LatticeType latticeType_ = LatticeType::Undefined;
    /// Communication buffers for MPI parallelism.
    helpme::vector<Complex> workSpace1_, workSpace2_;
    /// FFTW wrappers to help with transformations in the {A,B,C} dimensions.
    FFTWWrapper<Real> fftHelperA_, fftHelperB_, fftHelperC_;
    /// The cached list of splines.
    std::vector<SplineCacheEntry<Real>> splineCache_;
    /// A scratch array for each threads to use as storage when probing the grid.
    RealMat fractionalPhis_;
    /// A list of the splines that each thread should handle.
    std::vector<std::list<size_t>> splinesPerThread_;
    /// The transformation matrices for the compressed PME algorithms, in the {A,B,C} dimensions.
    RealMat compressionCoefficientsA_, compressionCoefficientsB_, compressionCoefficientsC_;
    /// Iterators that define the reciprocal lattice sums over each index, correctly defining -1/2 <= m{A,B,C} < 1/2.
    std::vector<int> mValsA_, mValsB_, mValsC_;
    /// A temporary list used in the assigning of atoms to threads and resorting by starting grid point.
    std::vector<std::set<std::pair<uint32_t, uint32_t>>> gridAtomList_;

    /*!
     * \brief makeGridIterator makes an iterator over the spline values that contribute to this node's grid
     *        in a given Cartesian dimension.  The iterator is of the form (grid point, spline index) and is
     *        sorted by increasing grid point, for cache efficiency.
     * \param dimension the dimension of the grid in the Cartesian dimension of interest.
     * \param first the first grid point in the Cartesian dimension to be handled by this node.
     * \param last the element past the last grid point in the Cartesian dimension to be handled by this node.
     * \param paddingSize the size of the "halo" region around this grid onto which the charge can be spread
     *        that really belongs to neighboring nodes.  For compressed PME we assume that each node handles
     *        only its own atoms and spreads onto an expanded grid to account for this.  In regular PME there
     *        is no padding because we assume that all halo atoms are present on this node before spreading.
     * \return the vector of spline iterators for each starting grid point.
     */
    GridIterator makeGridIterator(int dimension, int first, int last, int paddingSize) const {
        GridIterator gridIterator;
        if (paddingSize) {
            // This version assumes that every atom on this node is blindly place on the
            // grid, requiring that a padding area of size splineOrder-1 be present.
            for (int gridStart = 0; gridStart < dimension; ++gridStart) {
                std::vector<std::pair<short, short>> splineIterator(splineOrder_);
                splineIterator.clear();
                if (gridStart >= first && gridStart < last - paddingSize) {
                    for (int splineIndex = 0; splineIndex < splineOrder_; ++splineIndex) {
                        int gridPoint = (splineIndex + gridStart);
                        splineIterator.push_back(std::make_pair(gridPoint - first, splineIndex));
                    }
                }
                splineIterator.shrink_to_fit();
                gridIterator.push_back(splineIterator);
            }
        } else {
            // This version assumes that each node has its own atoms, plus "halo" atoms
            // from neighboring grids that can contribute to this node's grid.
            for (int gridStart = 0; gridStart < dimension; ++gridStart) {
                std::vector<std::pair<short, short>> splineIterator(splineOrder_);
                splineIterator.clear();
                for (int splineIndex = 0; splineIndex < splineOrder_; ++splineIndex) {
                    int gridPoint = (splineIndex + gridStart) % dimension;
                    if (gridPoint >= first && gridPoint < last)
                        splineIterator.push_back(std::make_pair(gridPoint - first, splineIndex));
                }
                splineIterator.shrink_to_fit();
                std::sort(splineIterator.begin(), splineIterator.end());
                gridIterator.push_back(splineIterator);
            }
        }
        gridIterator.shrink_to_fit();
        return gridIterator;
    }

    /*! Make sure that the iterator over AM components is up to date.
     * \param angMom the angular momentum required for the iterator over multipole components.
     */
    void updateAngMomIterator(int parameterAngMom) {
        auto L = parameterAngMom;
        size_t expectedNTerms = nCartesian(L);
        if (angMomIterator_.size() >= expectedNTerms) return;

        angMomIterator_.resize(expectedNTerms);
        for (int l = 0, count = 0; l <= L; ++l) {
            for (int lz = 0; lz <= l; ++lz) {
                for (int ly = 0; ly <= l - lz; ++ly) {
                    int lx = l - ly - lz;
                    angMomIterator_[count] = {{static_cast<short>(lx), static_cast<short>(ly), static_cast<short>(lz)}};
                    ++count;
                }
            }
        }
    }

    /*!
     * \brief updateInfluenceFunction builds the gF array cache, if the lattice vector has changed since the last
     *                                build of it.  If the cell is unchanged, this does nothing.  This is handled
     *                                separately from other initializations because we may skip the cacheing of
     *                                the influence function when the virial is requested; we assume it's an NPT
     *                                calculation in this case and therefore the influence function changes every time.
     */
    void updateInfluenceFunction() {
        if (unitCellHasChanged_ || kappaHasChanged_ || reciprocalSumDimensionHasChanged_ || splineOrderHasChanged_ ||
            scaleFactorHasChanged_ || numNodesHasChanged_ || algorithmHasChanged_) {
            cacheInfluenceFunctionFxn_(myNumKSumTermsA_, myNumKSumTermsB_, myNumKSumTermsC_, firstKSumTermA_,
                                       firstKSumTermB_, firstKSumTermC_, scaleFactor_, cachedInfluenceFunction_,
                                       recVecs_, cellVolume(), kappa_, &splineModA_[0], &splineModB_[0],
                                       &splineModC_[0], mValsA_.data(), mValsB_.data(), mValsC_.data(), nThreads_);
        }
    }

    /*!
     * \brief Runs a PME reciprocal space calculation, computing the potential and, optionally, its derivatives as
     *        well as the volume dependent part of the virial that comes from the structure factor.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.).  A negative value indicates that only the shell with |parameterAngMom| is to be considered,
     * e.g. a value of -2 specifies that only quadrupoles (and not dipoles or charges) will be provided; the input
     * matrix should have dimensions corresponding only to the number of terms in this shell.
     * \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
     * \param gridPoints the list of grid points at which the potential is needed; can be the same as the
     * coordinates.
     * \param derivativeLevel the order of the potential derivatives required; 0 is the potential, 1 is
     * (minus) the field, etc.  A negative value indicates that only the derivative with order |parameterAngMom|
     * is to be generated, e.g. -2 specifies that only the second derivative (not the potential or its gradient)
     * will be returned as output.  The output matrix should have space for only these terms, accordingly.
     * \param potential the array holding the potential.  This is a matrix of dimensions
     * nAtoms x nD, where nD is the derivative level requested.  See the details fo the parameters argument for
     * information about ordering of derivative components. N.B. this array is incremented with the potential, not
     * assigned, so take care to zero it first if only the current results are desired.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     */
    void computePRecHelper(int parameterAngMom, const RealMat &parameters, const RealMat &coordinates,
                           const RealMat &gridPoints, int derivativeLevel, RealMat &potential, RealMat &virial) {
        bool onlyOneShellForInput = parameterAngMom < 0;
        bool onlyOneShellForOutput = derivativeLevel < 0;
        parameterAngMom = std::abs(parameterAngMom);
        derivativeLevel = std::abs(derivativeLevel);
        int cartesianOffset = onlyOneShellForInput ? nCartesian(parameterAngMom - 1) : 0;
        sanityChecks(parameterAngMom, parameters, coordinates, cartesianOffset);
        updateAngMomIterator(std::max(parameterAngMom, derivativeLevel));
        // Note: we're calling the version of spread parameters that computes its own splines here.
        // This is quite inefficient, but allow the potential to be computed at arbitrary locations by
        // simply regenerating splines on demand in the probing stage.  If this becomes too slow, it's
        // easy to write some logic to check whether gridPoints and coordinates are the same, and
        // handle that special case using spline cacheing machinery for efficiency.
        Real *realGrid = reinterpret_cast<Real *>(workSpace1_.data());
        std::fill(workSpace1_.begin(), workSpace1_.end(), 0);
        updateAngMomIterator(parameterAngMom);
        auto fractionalParameters =
            cartesianTransform(parameterAngMom, onlyOneShellForInput, scaledRecVecs_.transpose(), parameters);
        int nComponents = nCartesian(parameterAngMom) - cartesianOffset;
        size_t nAtoms = coordinates.nRows();
        for (size_t atom = 0; atom < nAtoms; ++atom) {
            // Blindly reconstruct splines for this atom, assuming nothing about the validity of the cache.
            // Note that this incurs a somewhat steep cost due to repeated memory allocations.
            auto bSplines = makeBSplines(coordinates[atom], parameterAngMom);
            const auto &splineA = std::get<0>(bSplines);
            const auto &splineB = std::get<1>(bSplines);
            const auto &splineC = std::get<2>(bSplines);
            const auto &aGridIterator = gridIteratorA_[splineA.startingGridPoint()];
            const auto &bGridIterator = gridIteratorB_[splineB.startingGridPoint()];
            const auto &cGridIterator = gridIteratorC_[splineC.startingGridPoint()];
            int numPointsA = static_cast<int>(aGridIterator.size());
            int numPointsB = static_cast<int>(bGridIterator.size());
            int numPointsC = static_cast<int>(cGridIterator.size());
            const auto *iteratorDataA = aGridIterator.data();
            const auto *iteratorDataB = bGridIterator.data();
            const auto *iteratorDataC = cGridIterator.data();
            for (int component = 0; component < nComponents; ++component) {
                const auto &quanta = angMomIterator_[component + cartesianOffset];
                Real param = fractionalParameters(atom, component);
                const Real *splineValsA = splineA[quanta[0]];
                const Real *splineValsB = splineB[quanta[1]];
                const Real *splineValsC = splineC[quanta[2]];
                for (int pointC = 0; pointC < numPointsC; ++pointC) {
                    const auto &cPoint = iteratorDataC[pointC];
                    Real cValP = param * splineValsC[cPoint.second];
                    for (int pointB = 0; pointB < numPointsB; ++pointB) {
                        const auto &bPoint = iteratorDataB[pointB];
                        Real cbValP = cValP * splineValsB[bPoint.second];
                        Real *cbRow = &realGrid[cPoint.first * myGridDimensionB_ * myGridDimensionA_ +
                                                bPoint.first * myGridDimensionA_];
                        for (int pointA = 0; pointA < numPointsA; ++pointA) {
                            const auto &aPoint = iteratorDataA[pointA];
                            cbRow[aPoint.first] += cbValP * splineValsA[aPoint.second];
                        }
                    }
                }
            }
        }

        Real *potentialGrid;
        if (algorithmType_ == AlgorithmType::PME) {
            auto gridAddress = forwardTransform(realGrid);
            if (virial.nRows() == 0 && virial.nCols() == 0) {
                convolveE(gridAddress);
            } else {
                convolveEV(gridAddress, virial);
            }
            potentialGrid = inverseTransform(gridAddress);
        } else if (algorithmType_ == AlgorithmType::CompressedPME) {
            auto gridAddress = compressedForwardTransform(realGrid);
            if (virial.nRows() == 0 && virial.nCols() == 0) {
                convolveE(gridAddress);
                potentialGrid = compressedInverseTransform(gridAddress);
            } else {
                Real *convolvedGrid;
                convolveEV(gridAddress, convolvedGrid, virial);
                potentialGrid = compressedInverseTransform(convolvedGrid);
            }
        } else {
            std::logic_error("Unknown algorithm in helpme::computePRec");
        }

        auto fracPotential = potential.clone();
        fracPotential.setZero();
        cartesianOffset = onlyOneShellForOutput ? nCartesian(derivativeLevel - 1) : 0;
        int nPotentialComponents = nCartesian(derivativeLevel) - cartesianOffset;
        size_t nPoints = gridPoints.nRows();
        for (size_t point = 0; point < nPoints; ++point) {
            Real *phiPtr = fracPotential[point];
            auto bSplines = makeBSplines(gridPoints[point], derivativeLevel);
            auto splineA = std::get<0>(bSplines);
            auto splineB = std::get<1>(bSplines);
            auto splineC = std::get<2>(bSplines);
            const auto &aGridIterator = gridIteratorA_[splineA.startingGridPoint()];
            const auto &bGridIterator = gridIteratorB_[splineB.startingGridPoint()];
            const auto &cGridIterator = gridIteratorC_[splineC.startingGridPoint()];
            const Real *splineStartA = splineA[0];
            const Real *splineStartB = splineB[0];
            const Real *splineStartC = splineC[0];
            for (const auto &cPoint : cGridIterator) {
                for (const auto &bPoint : bGridIterator) {
                    const Real *cbRow = potentialGrid + cPoint.first * myGridDimensionA_ * myGridDimensionB_ +
                                        bPoint.first * myGridDimensionA_;
                    for (const auto &aPoint : aGridIterator) {
                        Real gridVal = cbRow[aPoint.first];
                        for (int component = 0; component < nPotentialComponents; ++component) {
                            const auto &quanta = angMomIterator_[component + cartesianOffset];
                            const Real *splineValsA = splineStartA + quanta[0] * splineOrder_;
                            const Real *splineValsB = splineStartB + quanta[1] * splineOrder_;
                            const Real *splineValsC = splineStartC + quanta[2] * splineOrder_;
                            phiPtr[component] += gridVal * splineValsA[aPoint.second] * splineValsB[bPoint.second] *
                                                 splineValsC[cPoint.second];
                        }
                    }
                }
            }
        }
        potential += cartesianTransform(derivativeLevel, onlyOneShellForOutput, scaledRecVecs_, fracPotential);
    }

    /*!
     * \brief Spreads parameters onto the grid for a single atom
     * \param atom the absolute atom number.
     * \param realGrid pointer to the array containing the grid in CBA order
     * \param nComponents the number of angular momentum components in the parameters.
     * \param nForceComponents the number of angular momentum components in the parameters with one extra
     *        level of angular momentum to permit evaluation of forces.
     * \param splineA the BSpline object for the A direction.
     * \param splineB the BSpline object for the B direction.
     * \param splineC the BSpline object for the C direction.
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
     * \param thread the ID of the thread handling this term.
     */
    void spreadParametersImpl(const int &atom, Real *realGrid, const int &nComponents, const Spline &splineA,
                              const Spline &splineB, const Spline &splineC, const RealMat &parameters, int thread) {
        const auto &aGridIterator = gridIteratorA_[splineA.startingGridPoint()];
        const auto &bGridIterator = gridIteratorB_[splineB.startingGridPoint()];
        const auto &cGridIterator = threadedGridIteratorC_[thread][splineC.startingGridPoint()];
        int numPointsA = static_cast<int>(aGridIterator.size());
        int numPointsB = static_cast<int>(bGridIterator.size());
        int numPointsC = static_cast<int>(cGridIterator.size());
        const auto *iteratorDataA = aGridIterator.data();
        const auto *iteratorDataB = bGridIterator.data();
        const auto *iteratorDataC = cGridIterator.data();
        for (int component = 0; component < nComponents; ++component) {
            const auto &quanta = angMomIterator_[component];
            Real param = parameters(atom, component);
            const Real *splineValsA = splineA[quanta[0]];
            const Real *splineValsB = splineB[quanta[1]];
            const Real *splineValsC = splineC[quanta[2]];
            for (int pointC = 0; pointC < numPointsC; ++pointC) {
                const auto &cPoint = iteratorDataC[pointC];
                Real cValP = param * splineValsC[cPoint.second];
                for (int pointB = 0; pointB < numPointsB; ++pointB) {
                    const auto &bPoint = iteratorDataB[pointB];
                    Real cbValP = cValP * splineValsB[bPoint.second];
                    Real *cbRow = realGrid + cPoint.first * myGridDimensionB_ * myGridDimensionA_ +
                                  bPoint.first * myGridDimensionA_;
                    for (int pointA = 0; pointA < numPointsA; ++pointA) {
                        const auto &aPoint = iteratorDataA[pointA];
                        cbRow[aPoint.first] += cbValP * splineValsA[aPoint.second];
                    }
                }
            }
        }
    }

    /*!
     * \brief Probes the grid and computes the force for a single atom, specialized for zero parameter angular momentum.
     * \param potentialGrid pointer to the array containing the potential, in ZYX order.
     * \param splineA the BSpline object for the A direction.
     * \param splineB the BSpline object for the B direction.
     * \param splineC the BSpline object for the C direction.
     * \param parameter the list of parameter associated with the given atom.
     * \param forces a 3 vector of the forces for this atom, ordered in memory as {Fx, Fy, Fz}.
     */
    void probeGridImpl(const Real *potentialGrid, const Spline &splineA, const Spline &splineB, const Spline &splineC,
                       const Real &parameter, Real *forces) const {
        const auto &aGridIterator = gridIteratorA_[splineA.startingGridPoint()];
        const auto &bGridIterator = gridIteratorB_[splineB.startingGridPoint()];
        const auto &cGridIterator = gridIteratorC_[splineC.startingGridPoint()];
        // We unpack the vector to raw pointers, as profiling shows that using range based for loops over vectors
        // causes a signficant penalty in the innermost loop, primarily due to checking the loop stop condition.
        int numPointsA = static_cast<int>(aGridIterator.size());
        int numPointsB = static_cast<int>(bGridIterator.size());
        int numPointsC = static_cast<int>(cGridIterator.size());
        const auto *iteratorDataA = aGridIterator.data();
        const auto *iteratorDataB = bGridIterator.data();
        const auto *iteratorDataC = cGridIterator.data();
        const Real *splineStartA0 = splineA[0];
        const Real *splineStartB0 = splineB[0];
        const Real *splineStartC0 = splineC[0];
        const Real *splineStartA1 = splineStartA0 + splineOrder_;
        const Real *splineStartB1 = splineStartB0 + splineOrder_;
        const Real *splineStartC1 = splineStartC0 + splineOrder_;
        Real Ex = 0, Ey = 0, Ez = 0;
        for (int pointC = 0; pointC < numPointsC; ++pointC) {
            const auto &cPoint = iteratorDataC[pointC];
            const Real &splineC0 = splineStartC0[cPoint.second];
            const Real &splineC1 = splineStartC1[cPoint.second];
            for (int pointB = 0; pointB < numPointsB; ++pointB) {
                const auto &bPoint = iteratorDataB[pointB];
                const Real &splineB0 = splineStartB0[bPoint.second];
                const Real &splineB1 = splineStartB1[bPoint.second];
                const Real *cbRow = potentialGrid + cPoint.first * myGridDimensionA_ * myGridDimensionB_ +
                                    bPoint.first * myGridDimensionA_;
                for (int pointA = 0; pointA < numPointsA; ++pointA) {
                    const auto &aPoint = iteratorDataA[pointA];
                    const Real &splineA0 = splineStartA0[aPoint.second];
                    const Real &splineA1 = splineStartA1[aPoint.second];
                    const Real &gridVal = cbRow[aPoint.first];
                    Ey += gridVal * splineA0 * splineB1 * splineC0;
                    Ez += gridVal * splineA0 * splineB0 * splineC1;
                    Ex += gridVal * splineA1 * splineB0 * splineC0;
                }
            }
        }

        forces[0] -= parameter * (scaledRecVecs_[0][0] * Ex + scaledRecVecs_[0][1] * Ey + scaledRecVecs_[0][2] * Ez);
        forces[1] -= parameter * (scaledRecVecs_[1][0] * Ex + scaledRecVecs_[1][1] * Ey + scaledRecVecs_[1][2] * Ez);
        forces[2] -= parameter * (scaledRecVecs_[2][0] * Ex + scaledRecVecs_[2][1] * Ey + scaledRecVecs_[2][2] * Ez);
    }

    /*!
     * \brief Probes the grid and computes the force for a single atom, for arbitrary parameter angular momentum.
     * \param potentialGrid pointer to the array containing the potential, in ZYX order.
     * \param nPotentialComponents the number of components in the potential and its derivatives with one extra
     *        level of angular momentum to permit evaluation of forces.
     * \param splineA the BSpline object for the A direction.
     * \param splineB the BSpline object for the B direction.
     * \param splineC the BSpline object for the C direction.
     * \param phiPtr a scratch array of length nPotentialComponents, to store the fractional potential.
     * N.B. Make sure that updateAngMomIterator() has been called first with the appropriate derivative
     * level for the requested potential derivatives.
     */
    void probeGridImpl(const Real *potentialGrid, const int &nPotentialComponents, const Spline &splineA,
                       const Spline &splineB, const Spline &splineC, Real *phiPtr) {
        const auto &aGridIterator = gridIteratorA_[splineA.startingGridPoint()];
        const auto &bGridIterator = gridIteratorB_[splineB.startingGridPoint()];
        const auto &cGridIterator = gridIteratorC_[splineC.startingGridPoint()];
        const Real *splineStartA = splineA[0];
        const Real *splineStartB = splineB[0];
        const Real *splineStartC = splineC[0];
        for (const auto &cPoint : cGridIterator) {
            for (const auto &bPoint : bGridIterator) {
                const Real *cbRow = potentialGrid + cPoint.first * myGridDimensionA_ * myGridDimensionB_ +
                                    bPoint.first * myGridDimensionA_;
                for (const auto &aPoint : aGridIterator) {
                    Real gridVal = cbRow[aPoint.first];
                    for (int component = 0; component < nPotentialComponents; ++component) {
                        const auto &quanta = angMomIterator_[component];
                        const Real *splineValsA = splineStartA + quanta[0] * splineOrder_;
                        const Real *splineValsB = splineStartB + quanta[1] * splineOrder_;
                        const Real *splineValsC = splineStartC + quanta[2] * splineOrder_;
                        phiPtr[component] += gridVal * splineValsA[aPoint.second] * splineValsB[bPoint.second] *
                                             splineValsC[cPoint.second];
                    }
                }
            }
        }
    }

    /*!
     * \brief Probes the grid and computes the force for a single atom, for arbitrary parameter angular momentum.
     * \param atom the absolute atom number.
     * \param potentialGrid pointer to the array containing the potential, in ZYX order.
     * \param nComponents the number of angular momentum components in the parameters.
     * \param nForceComponents the number of angular momentum components in the parameters with one extra
     *        level of angular momentum to permit evaluation of forces.
     * \param splineA the BSpline object for the A direction.
     * \param splineB the BSpline object for the B direction.
     * \param splineC the BSpline object for the C direction.
     * \param phiPtr a scratch array of length nForceComponents, to store the fractional potential.
     * \param fracParameters the list of parameters associated with the current atom, in
     * the scaled fraction coordinate basis (charges, C6 coefficients,
     * multipoles, etc...). For a parameter with angular momentum L, a matrix
     * of dimension nAtoms x nL is expected, where
     * nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     */
    void probeGridImpl(const int &atom, const Real *potentialGrid, const int &nComponents, const int &nForceComponents,
                       const Spline &splineA, const Spline &splineB, const Spline &splineC, Real *phiPtr,
                       const Real *fracParameters, Real *forces) {
        std::fill(phiPtr, phiPtr + nForceComponents, 0);
        probeGridImpl(potentialGrid, nForceComponents, splineA, splineB, splineC, phiPtr);

        Real fracForce[3] = {0, 0, 0};
        for (int component = 0; component < nComponents; ++component) {
            Real param = fracParameters[component];
            const auto &quanta = angMomIterator_[component];
            short lx = quanta[0];
            short ly = quanta[1];
            short lz = quanta[2];
            fracForce[0] -= param * phiPtr[cartAddress(lx + 1, ly, lz)];
            fracForce[1] -= param * phiPtr[cartAddress(lx, ly + 1, lz)];
            fracForce[2] -= param * phiPtr[cartAddress(lx, ly, lz + 1)];
        }
        forces[0] += scaledRecVecs_[0][0] * fracForce[0] + scaledRecVecs_[0][1] * fracForce[1] +
                     scaledRecVecs_[0][2] * fracForce[2];
        forces[1] += scaledRecVecs_[1][0] * fracForce[0] + scaledRecVecs_[1][1] * fracForce[1] +
                     scaledRecVecs_[1][2] * fracForce[2];
        forces[2] += scaledRecVecs_[2][0] * fracForce[0] + scaledRecVecs_[2][1] * fracForce[1] +
                     scaledRecVecs_[2][2] * fracForce[2];
    }

    /*!
     * \brief assertInitialized makes sure that setup() has been called before running any calculations.
     */
    void assertInitialized() const {
        if (!rPower_)
            throw std::runtime_error(
                "Either setup(...) or setup_parallel(...) must be called before computing anything.");
    }

    /*!
     * \brief makeBSplines construct the {x,y,z} B-Splines.
     * \param atomCoords a 3-vector containing the atom's coordinates.
     * \param derivativeLevel level of derivative needed for the splines.
     * \return a 3-tuple containing the {x,y,z} B-splines.
     */
    std::tuple<Spline, Spline, Spline> makeBSplines(const Real *atomCoords, short derivativeLevel) const {
        // Subtract a tiny amount to make sure we're not exactly on the rightmost (excluded)
        // grid point. The calculation is translationally invariant, so this is valid.
        constexpr float EPS = 1e-6f;
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
        short aStartingGridPoint = gridDimensionA_ * aCoord;
        short bStartingGridPoint = gridDimensionB_ * bCoord;
        short cStartingGridPoint = gridDimensionC_ * cCoord;
        Real aDistanceFromGridPoint = gridDimensionA_ * aCoord - aStartingGridPoint;
        Real bDistanceFromGridPoint = gridDimensionB_ * bCoord - bStartingGridPoint;
        Real cDistanceFromGridPoint = gridDimensionC_ * cCoord - cStartingGridPoint;
        return std::make_tuple(Spline(aStartingGridPoint, aDistanceFromGridPoint, splineOrder_, derivativeLevel),
                               Spline(bStartingGridPoint, bDistanceFromGridPoint, splineOrder_, derivativeLevel),
                               Spline(cStartingGridPoint, cDistanceFromGridPoint, splineOrder_, derivativeLevel));
    }

    /*!
     * \brief convolveEVImpl performs the reciprocal space convolution, returning the energy, for conventional PME.
     *       We opt to not cache this the same way as the non-virial version because it's safe to assume that if
     *       the virial is requested the box is likely to change, which renders the cache useless.
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param useConjugateSymmetry whether to use the complex conjugate symmetry in the convolution or not.
     * \param fullNx full (complex) dimension of the reciprocal sum in the X direction.
     * \param myNx the subset of the reciprocal sum in the x direction to be handled by this node.
     * \param myNy the subset of the reciprocal sum in the y direction to be handled by this node.
     * \param myNz the subset of the reciprocal sum in the z direction to be handled by this node.
     * \param startX the starting reciprocal sum term handled by this node in the X direction.
     * \param startY the starting reciprocal sum term handled by this node in the Y direction.
     * \param startZ the starting reciprocal sum term handled by this node in the Z direction.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param gridPtr the Fourier space grid, with ordering YXZ.
     * \param boxInv the reciprocal lattice vectors.
     * \param volume the volume of the unit cell.
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param xMods the Fourier space norms of the x B-Splines.
     * \param yMods the Fourier space norms of the y B-Splines.
     * \param zMods the Fourier space norms of the z B-Splines.
     * \param xMVals the integer prefactors to iterate over reciprocal vectors in the x dimension.
     * \param yMVals the integer prefactors to iterate over reciprocal vectors in the y dimension.
     * \param zMVals the integer prefactors to iterate over reciprocal vectors in the z dimension.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \param nThreads the number of OpenMP threads to use.
     * \return the reciprocal space energy.
     */
    template <int rPower>
    static Real convolveEVImpl(bool useConjugateSymmetry, int fullNx, int myNx, int myNy, int myNz, int startX,
                               int startY, int startZ, Real scaleFactor, Complex *gridPtr, const RealMat &boxInv,
                               Real volume, Real kappa, const Real *xMods, const Real *yMods, const Real *zMods,
                               const int *xMVals, const int *yMVals, const int *zMVals, RealMat &virial, int nThreads) {
        Real energy = 0;

        bool nodeZero = startX == 0 && startY == 0 && startZ == 0;
        if (rPower > 3 && nodeZero) {
            // Kernels with rPower>3 are absolutely convergent and should have the m=0 term present.
            // To compute it we need sum_ij c(i)c(j), which can be obtained from the structure factor norm.
            Real prefac = 2 * scaleFactor * HELPME_PI * HELPME_SQRTPI * pow(kappa, rPower - 3) /
                          ((rPower - 3) * gammaComputer<Real, rPower>::value * volume);
            energy += prefac * (gridPtr[0].real() * gridPtr[0].real() + gridPtr[0].imag() * gridPtr[0].imag());
        }
        // Ensure the m=0 term convolution product is zeroed for the backtransform; it's been accounted for above.
        if (nodeZero) gridPtr[0] = Complex(0, 0);

        Real bPrefac = HELPME_PI * HELPME_PI / (kappa * kappa);
        Real volPrefac =
            scaleFactor * pow(HELPME_PI, rPower - 1) / (HELPME_SQRTPI * gammaComputer<Real, rPower>::value * volume);
        size_t nxz = (size_t)myNx * myNz;
        Real Vxx = 0, Vxy = 0, Vyy = 0, Vxz = 0, Vyz = 0, Vzz = 0;
        const Real *boxPtr = boxInv[0];
        size_t nyxz = myNy * nxz;
        // Exclude m=0 cell.
        int start = (nodeZero ? 1 : 0);
// Writing the three nested loops in one allows for better load balancing in parallel.
#pragma omp parallel for reduction(+ : energy, Vxx, Vxy, Vyy, Vxz, Vyz, Vzz) num_threads(nThreads)
        for (size_t yxz = start; yxz < nyxz; ++yxz) {
            size_t xz = yxz % nxz;
            short ky = yxz / nxz;
            short kx = xz / myNz;
            short kz = xz % myNz;
            // We only loop over the first nx/2+1 x values in the complex case;
            // this accounts for the "missing" complex conjugate values.
            Real permPrefac = (useConjugateSymmetry && (kx + startX != 0) && (kx + startX != fullNx - 1)) ? 2 : 1;
            const int &mx = xMVals[kx];
            const int &my = yMVals[ky];
            const int &mz = zMVals[kz];
            Real mVecX = boxPtr[0] * mx + boxPtr[1] * my + boxPtr[2] * mz;
            Real mVecY = boxPtr[3] * mx + boxPtr[4] * my + boxPtr[5] * mz;
            Real mVecZ = boxPtr[6] * mx + boxPtr[7] * my + boxPtr[8] * mz;
            Real mNormSq = mVecX * mVecX + mVecY * mVecY + mVecZ * mVecZ;
            Real mTerm = raiseNormToIntegerPower<Real, rPower - 3>::compute(mNormSq);
            Real bSquared = bPrefac * mNormSq;
            auto gammas = incompleteGammaVirialComputer<Real, 3 - rPower>::compute(bSquared);
            Real eGamma = std::get<0>(gammas);
            Real vGamma = std::get<1>(gammas);
            Complex &gridVal = gridPtr[yxz];
            Real structFacNorm = gridVal.real() * gridVal.real() + gridVal.imag() * gridVal.imag();
            Real totalPrefac = volPrefac * mTerm * yMods[ky] * xMods[kx] * zMods[kz];
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
     * \brief convolveEVCompressedImpl performs the reciprocal space convolution, returning the energy, for compressed
     * PME. We opt to not cache this the same way as the non-virial version because it's safe to assume that if the
     * virial is requested the box is likely to change, which renders the cache useless.
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param myNx the subset of the reciprocal sum in the x direction to be handled by this node.
     * \param myNy the subset of the reciprocal sum in the y direction to be handled by this node.
     * \param myNz the subset of the reciprocal sum in the z direction to be handled by this node.
     * \param startX the starting reciprocal sum term handled by this node in the X direction.
     * \param startY the starting reciprocal sum term handled by this node in the Y direction.
     * \param startZ the starting reciprocal sum term handled by this node in the Z direction.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof
     *  (e.g. thee 1 / [4 pi epslion0] for Coulomb calculations).
     * \param gridPtrIn the Fourier space grid, with ordering YXZ.
     * \param gridPtrOut the convolved Fourier space grid, with ordering YXZ.
     * \param boxInv the reciprocal lattice vectors.
     * \param volume the volume of the unit cell.
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param xMods the Fourier space norms of the x B-Splines.
     * \param yMods the Fourier space norms of the y B-Splines.
     * \param zMods the Fourier space norms of the z B-Splines.
     * \param xMVals the integer prefactors to iterate over reciprocal vectors in the x dimension.
     * \param yMVals the integer prefactors to iterate over reciprocal vectors in the y dimension.
     * \param zMVals the integer prefactors to iterate over reciprocal vectors in the z dimension.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \param nThreads the number of OpenMP threads to use.
     * \return the reciprocal space energy.
     */
    template <int rPower>
    static Real convolveEVCompressedImpl(int myNx, int myNy, int myNz, int startX, int startY, int startZ,
                                         Real scaleFactor, const Real *__restrict__ gridPtrIn,
                                         Real *__restrict__ gridPtrOut, const RealMat &boxInv, Real volume, Real kappa,
                                         const Real *xMods, const Real *yMods, const Real *zMods, const int *xMVals,
                                         const int *yMVals, const int *zMVals, RealMat &virial, int nThreads) {
        Real energy = 0;

        bool nodeZero = startX == 0 && startY == 0 && startZ == 0;
        if (rPower > 3 && nodeZero) {
            // Kernels with rPower>3 are absolutely convergent and should have the m=0 term present.
            // To compute it we need sum_ij c(i)c(j), which can be obtained from the structure factor norm.
            Real prefac = 2 * scaleFactor * HELPME_PI * HELPME_SQRTPI * pow(kappa, rPower - 3) /
                          ((rPower - 3) * gammaComputer<Real, rPower>::value * volume);
            energy += prefac * gridPtrIn[0] * gridPtrIn[0];
        }
        // Ensure the m=0 term convolution product is zeroed for the backtransform; it's been accounted for above.
        if (nodeZero) gridPtrOut[0] = 0;

        Real bPrefac = HELPME_PI * HELPME_PI / (kappa * kappa);
        Real volPrefac =
            scaleFactor * pow(HELPME_PI, rPower - 1) / (HELPME_SQRTPI * gammaComputer<Real, rPower>::value * volume);
        size_t nxz = (size_t)myNx * myNz;
        size_t nyxz = myNy * nxz;
        Real Vxx = 0, Vxy = 0, Vyy = 0, Vxz = 0, Vyz = 0, Vzz = 0;
        const Real *boxPtr = boxInv[0];
        // Exclude m=0 cell.
        int start = (nodeZero ? 1 : 0);
// Writing the three nested loops in one allows for better load balancing in parallel.
#pragma omp parallel for reduction(+ : energy, Vxx, Vxy, Vyy, Vxz, Vyz, Vzz) num_threads(nThreads)
        for (size_t yxz = start; yxz < nyxz; ++yxz) {
            size_t xz = yxz % nxz;
            short ky = yxz / nxz;
            short kx = xz / myNz;
            short kz = xz % myNz;
            // We only loop over the first nx/2+1 x values in the complex case;
            // this accounts for the "missing" complex conjugate values.
            const int &mx = xMVals[kx];
            const int &my = yMVals[ky];
            const int &mz = zMVals[kz];
            Real mVecX = boxPtr[0] * mx + boxPtr[1] * my + boxPtr[2] * mz;
            Real mVecY = boxPtr[3] * mx + boxPtr[4] * my + boxPtr[5] * mz;
            Real mVecZ = boxPtr[6] * mx + boxPtr[7] * my + boxPtr[8] * mz;
            Real mNormSq = mVecX * mVecX + mVecY * mVecY + mVecZ * mVecZ;
            Real mTerm = raiseNormToIntegerPower<Real, rPower - 3>::compute(mNormSq);
            Real bSquared = bPrefac * mNormSq;
            auto gammas = incompleteGammaVirialComputer<Real, 3 - rPower>::compute(bSquared);
            Real eGamma = std::get<0>(gammas);
            Real vGamma = std::get<1>(gammas);
            const Real &gridVal = gridPtrIn[yxz];
            size_t minusKx = (mx == 0 ? 0 : (mx < 0 ? kx - 1 : kx + 1));
            size_t minusKy = (my == 0 ? 0 : (my < 0 ? ky - 1 : ky + 1));
            size_t minusKz = (mz == 0 ? 0 : (mz < 0 ? kz - 1 : kz + 1));
            size_t addressXY = minusKy * nxz + minusKx * myNz + kz;
            size_t addressXZ = ky * nxz + minusKx * myNz + minusKz;
            size_t addressYZ = minusKy * nxz + (size_t)kx * myNz + minusKz;
            Real totalPrefac = volPrefac * mTerm * yMods[ky] * xMods[kx] * zMods[kz];
            Real influenceFunction = totalPrefac * eGamma;
            gridPtrOut[yxz] = gridVal * influenceFunction;
            Real eTerm = influenceFunction * gridVal * gridVal;
            Real vPrefac = vGamma * totalPrefac / mNormSq * gridVal;
            Real vTerm = vPrefac * gridVal;
            Real vTermXY = vPrefac * gridPtrIn[addressXY];
            Real vTermXZ = vPrefac * gridPtrIn[addressXZ];
            Real vTermYZ = vPrefac * gridPtrIn[addressYZ];
            energy += eTerm;
            Vxx += vTerm * mVecX * mVecX;
            Vxy -= vTermXY * mVecX * mVecY;
            Vyy += vTerm * mVecY * mVecY;
            Vxz -= vTermXZ * mVecX * mVecZ;
            Vyz -= vTermYZ * mVecY * mVecZ;
            Vzz += vTerm * mVecZ * mVecZ;
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
     * \brief checkMinimumImageCutoff ensure that the box dimensions satisfy the condition
     *       sphericalCutoff < MIN(W_A, W_B, W_C)/2
     *
     *       where
     *
     *       W_A = |A.(B x C)| / |B x C|
     *       W_B = |B.(C x A)| / |C x A|
     *       W_C = |C.(A x B)| / |A x B|
     *
     * \param sphericalCutoff the spherical nonbonded cutoff in Angstrom
     */
    void checkMinimumImageCutoff(int sphericalCutoff) {
        Real V = cellVolume();
        Real ABx = boxVecs_(0, 1) * boxVecs_(1, 2) - boxVecs_(0, 2) * boxVecs_(1, 1);
        Real ABy = boxVecs_(0, 0) * boxVecs_(1, 2) - boxVecs_(0, 2) * boxVecs_(1, 0);
        Real ABz = boxVecs_(0, 0) * boxVecs_(1, 1) - boxVecs_(0, 1) * boxVecs_(1, 0);
        Real ACx = boxVecs_(0, 1) * boxVecs_(2, 2) - boxVecs_(0, 2) * boxVecs_(2, 1);
        Real ACy = boxVecs_(0, 0) * boxVecs_(2, 2) - boxVecs_(0, 2) * boxVecs_(2, 0);
        Real ACz = boxVecs_(0, 0) * boxVecs_(2, 1) - boxVecs_(0, 1) * boxVecs_(2, 0);
        Real BCx = boxVecs_(1, 1) * boxVecs_(2, 2) - boxVecs_(1, 2) * boxVecs_(2, 1);
        Real BCy = boxVecs_(1, 0) * boxVecs_(2, 2) - boxVecs_(1, 2) * boxVecs_(2, 0);
        Real BCz = boxVecs_(1, 0) * boxVecs_(2, 1) - boxVecs_(1, 1) * boxVecs_(2, 0);
        Real AxBnorm = std::sqrt(ABx * ABx + ABy * ABy + ABz * ABz);
        Real AxCnorm = std::sqrt(ACx * ACx + ACy * ACy + ACz * ACz);
        Real BxCnorm = std::sqrt(BCx * BCx + BCy * BCy + BCz * BCz);
        Real minDim = 2 * sphericalCutoff;
        if (V / AxBnorm < minDim || V / AxCnorm < minDim || V / BxCnorm < minDim)
            throw std::runtime_error("The cutoff used must be less than half of the minimum of three box widths");
    }

    /*!
     * \brief sanityChecks just makes sure that inputs have consistent dimensions, and that prerequisites are
     * initialized.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.).
     * \param parameters the input parameters.
     * \param coordinates the input coordinates.
     * \param cartesianOffset an offset to the start of the angular momentum shell for the parameters, in cases where
     * only a single angular momentum shell is to be processed (rather than all shells up to a given angular momentum).
     */
    void sanityChecks(int parameterAngMom, const RealMat &parameters, const RealMat &coordinates,
                      int cartesianOffset = 0) {
        assertInitialized();
        if (parameterAngMom < 0)
            throw std::runtime_error("Negative parameter angular momentum found where positive value was expected");
        if (boxVecs_.isNearZero())
            throw std::runtime_error(
                "Lattice vectors have not been set yet!  Call setLatticeVectors(...) before runPME(...);");
        if (coordinates.nRows() != parameters.nRows())
            throw std::runtime_error(
                "Inconsistent number of coordinates and parameters; there should be nAtoms of each.");
        if (parameters.nCols() != (nCartesian(parameterAngMom) - cartesianOffset))
            throw std::runtime_error(
                "Mismatch in the number of parameters provided and the parameter angular momentum");
    }

    /*!
     * \brief cacheInfluenceFunctionImpl computes the influence function used in convolution, for later use.
     * \tparam rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive dispersion).
     * \param myNx the subset of the grid in the x direction to be handled by this node.
     * \param myNy the subset of the grid in the y direction to be handled by this node.
     * \param myNz the subset of the grid in the z direction to be handled by this node.
     * \param startX the starting reciprocal space sum term handled by this node in the X direction.
     * \param startY the starting reciprocal space sum term handled by this node in the Y direction.
     * \param startZ the starting reciprocal space sum term handled by this node in the Z direction.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param gridPtr the Fourier space grid, with ordering YXZ.
     * \param boxInv the reciprocal lattice vectors.
     * \param volume the volume of the unit cell.
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param xMods the Fourier space norms of the x B-Splines.
     * \param yMods the Fourier space norms of the y B-Splines.
     * \param zMods the Fourier space norms of the z B-Splines.
     * \param xMVals the integer prefactors to iterate over reciprocal vectors in the x dimension.
     * \param yMVals the integer prefactors to iterate over reciprocal vectors in the y dimension.
     * \param zMVals the integer prefactors to iterate over reciprocal vectors in the z dimension.
     *        This vector is incremented, not assigned.
     * \param nThreads the number of OpenMP threads to use.
     * \return the energy for the m=0 term.
     */
    template <int rPower>
    static void cacheInfluenceFunctionImpl(int myNx, int myNy, int myNz, int startX, int startY, int startZ,
                                           Real scaleFactor, RealVec &influenceFunction, const RealMat &boxInv,
                                           Real volume, Real kappa, const Real *xMods, const Real *yMods,
                                           const Real *zMods, const int *xMVals, const int *yMVals, const int *zMVals,
                                           int nThreads) {
        bool nodeZero = startX == 0 && startY == 0 && startZ == 0;
        size_t nxz = (size_t)myNx * myNz;
        size_t nyxz = myNy * nxz;
        influenceFunction.resize(nyxz);
        Real *gridPtr = influenceFunction.data();
        if (nodeZero) gridPtr[0] = 0;

        Real bPrefac = HELPME_PI * HELPME_PI / (kappa * kappa);
        Real volPrefac =
            scaleFactor * pow(HELPME_PI, rPower - 1) / (HELPME_SQRTPI * gammaComputer<Real, rPower>::value * volume);
        const Real *boxPtr = boxInv[0];
        // Exclude m=0 cell.
        int start = (nodeZero ? 1 : 0);
// Writing the three nested loops in one allows for better load balancing in parallel.
#pragma omp parallel for num_threads(nThreads)
        for (size_t yxz = start; yxz < nyxz; ++yxz) {
            size_t xz = yxz % nxz;
            short ky = yxz / nxz;
            short kx = xz / myNz;
            short kz = xz % myNz;
            const Real mx = (Real)xMVals[kx];
            const Real my = (Real)yMVals[ky];
            const Real mz = (Real)zMVals[kz];
            Real mVecX = boxPtr[0] * mx + boxPtr[1] * my + boxPtr[2] * mz;
            Real mVecY = boxPtr[3] * mx + boxPtr[4] * my + boxPtr[5] * mz;
            Real mVecZ = boxPtr[6] * mx + boxPtr[7] * my + boxPtr[8] * mz;
            Real mNormSq = mVecX * mVecX + mVecY * mVecY + mVecZ * mVecZ;
            Real mTerm = raiseNormToIntegerPower<Real, rPower - 3>::compute(mNormSq);
            Real bSquared = bPrefac * mNormSq;
            Real incompleteGammaTerm = incompleteGammaComputer<Real, 3 - rPower>::compute(bSquared);
            gridPtr[yxz] = volPrefac * incompleteGammaTerm * mTerm * yMods[ky] * xMods[kx] * zMods[kz];
        }
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
     * \brief slfEImpl computes the coefficient to be applied to the sum of squared parameters for the self energy
     *                 due to particles feeling their own potential.
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
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof
     *        (e.g. the 1 / [4 pi epslion0] for Coulomb calculations).
     * \return the coefficient for the sum of squared parameters in the self energy.  N.B. there is no self force
     * associated with this term.
     */
    template <int rPower>
    static Real slfEImpl(int parameterAngMom, Real kappa, Real scaleFactor) {
        if (parameterAngMom) throw std::runtime_error("Multipole self terms have not been coded yet.");
        return -scaleFactor * std::pow(kappa, rPower) / (rPower * gammaComputer<Real, rPower>::value);
    }

    /*!
     * \brief common_init sets up information that is common to serial and parallel runs.
     */
    void setupCalculationMetadata(int rPower, Real kappa, int splineOrder, int dimA, int dimB, int dimC, int maxKA,
                                  int maxKB, int maxKC, Real scaleFactor, int nThreads, void *commPtrIn,
                                  NodeOrder nodeOrder, int numNodesA, int numNodesB, int numNodesC) {
        int numKSumTermsA = std::min(2 * maxKA + 1, dimA);
        int numKSumTermsB = std::min(2 * maxKB + 1, dimB);
        int numKSumTermsC = std::min(2 * maxKC + 1, dimC);
        AlgorithmType algorithm = numKSumTermsA < dimA && numKSumTermsB < dimB && numKSumTermsC < dimC
                                      ? AlgorithmType::CompressedPME
                                      : AlgorithmType::PME;
        kappaHasChanged_ = kappa != kappa_;
        numNodesHasChanged_ = numNodesA_ != numNodesA || numNodesB_ != numNodesB || numNodesC_ != numNodesC;
        rPowerHasChanged_ = rPower_ != rPower;
        gridDimensionHasChanged_ = gridDimensionA_ != dimA || gridDimensionB_ != dimB || gridDimensionC_ != dimC;
        reciprocalSumDimensionHasChanged_ =
            numKSumTermsA != numKSumTermsA_ || numKSumTermsB != numKSumTermsB_ || numKSumTermsC != numKSumTermsC_;
        algorithmHasChanged_ = algorithmType_ != algorithm;
        splineOrderHasChanged_ = splineOrder_ != splineOrder;
        scaleFactorHasChanged_ = scaleFactor_ != scaleFactor;
        if (kappaHasChanged_ || rPowerHasChanged_ || gridDimensionHasChanged_ || splineOrderHasChanged_ ||
            numNodesHasChanged_ || scaleFactorHasChanged_ || algorithmHasChanged_ ||
            requestedNumberOfThreads_ != nThreads) {
            numNodesA_ = numNodesA;
            numNodesB_ = numNodesB;
            numNodesC_ = numNodesC;
            myNodeRankA_ = myNodeRankB_ = myNodeRankC_ = 0;
#if HAVE_MPI == 1
            if (commPtrIn) {
                MPI_Comm const &communicator = *((MPI_Comm *)(commPtrIn));
                mpiCommunicator_ = std::unique_ptr<MPIWrapper<Real>>(
                    new MPIWrapper<Real>(communicator, numNodesA, numNodesB, numNodesC));
                switch (nodeOrder) {
                    case (NodeOrder::ZYX):
                        myNodeRankA_ = mpiCommunicator_->myRank_ % numNodesA;
                        myNodeRankB_ = (mpiCommunicator_->myRank_ % (numNodesB * numNodesA)) / numNodesA;
                        myNodeRankC_ = mpiCommunicator_->myRank_ / (numNodesB * numNodesA);
                        mpiCommunicatorA_ =
                            mpiCommunicator_->split(myNodeRankC_ * numNodesB + myNodeRankB_, myNodeRankA_);
                        mpiCommunicatorB_ =
                            mpiCommunicator_->split(myNodeRankC_ * numNodesA + myNodeRankA_, myNodeRankB_);
                        mpiCommunicatorC_ =
                            mpiCommunicator_->split(myNodeRankB_ * numNodesA + myNodeRankA_, myNodeRankC_);
                        break;
                    default:
                        throw std::runtime_error("Unknown NodeOrder in helpme::setupCalculationMetadata.");
                }
            }
#else   // Have MPI
            if (numNodesA * numNodesB * numNodesC > 1)
                throw std::runtime_error(
                    "a parallel calculation has been setup, but helpme was not compiled with MPI.  Make sure you "
                    "compile with -DHAVE_MPI=1 "
                    "in the list of compiler definitions.");
#endif  // Have MPI
            rPower_ = rPower;
            algorithmType_ = algorithm;
            splineOrder_ = splineOrder;
            cacheLineSizeInReals_ = static_cast<Real>(sysconf(_SC_PAGESIZE) / sizeof(Real));
            requestedNumberOfThreads_ = nThreads;
#ifdef _OPENMP
            nThreads_ = nThreads ? nThreads : omp_get_max_threads();
#else
            nThreads_ = 1;
#endif
            scaleFactor_ = scaleFactor;
            kappa_ = kappa;

            size_t scratchSize;
            int gridPaddingA = 0, gridPaddingB = 0, gridPaddingC = 0;
            if (algorithm == AlgorithmType::CompressedPME) {
                gridDimensionA_ = numNodesA * std::ceil(dimA / (float)numNodesA);
                gridDimensionB_ = numNodesB * std::ceil(dimB / (float)numNodesB);
                gridDimensionC_ = numNodesC * std::ceil(dimC / (float)numNodesC);
                gridPaddingA = (numNodesA > 1 ? splineOrder - 1 : 0);
                gridPaddingB = (numNodesB > 1 ? splineOrder - 1 : 0);
                gridPaddingC = (numNodesC > 1 ? splineOrder - 1 : 0);
                myGridDimensionA_ = gridDimensionA_ / numNodesA + gridPaddingA;
                myGridDimensionB_ = gridDimensionB_ / numNodesB + gridPaddingB;
                myGridDimensionC_ = gridDimensionC_ / numNodesC + gridPaddingC;
                myFirstGridPointA_ = myNodeRankA_ * (myGridDimensionA_ - gridPaddingA);
                myFirstGridPointB_ = myNodeRankB_ * (myGridDimensionB_ - gridPaddingB);
                myFirstGridPointC_ = myNodeRankC_ * (myGridDimensionC_ - gridPaddingC);
                myNumKSumTermsA_ = numNodesA == 1 ? numKSumTermsA : 2 * std::ceil((maxKA + 1.0) / numNodesA);
                myNumKSumTermsB_ = numNodesB == 1 ? numKSumTermsB : 2 * std::ceil((maxKB + 1.0) / numNodesB);
                myNumKSumTermsC_ = numNodesC == 1 ? numKSumTermsC : 2 * std::ceil((maxKC + 1.0) / numNodesC);
                numKSumTermsA_ = myNumKSumTermsA_ * numNodesA;
                numKSumTermsB_ = myNumKSumTermsB_ * numNodesB;
                numKSumTermsC_ = myNumKSumTermsC_ * numNodesC;
                firstKSumTermA_ = myNodeRankA_ * myNumKSumTermsA_;
                firstKSumTermB_ = myNodeRankB_ * myNumKSumTermsB_;
                firstKSumTermC_ = myNodeRankC_ * myNumKSumTermsC_;
                fftHelperA_ = std::move(FFTWWrapper<Real>());
                fftHelperB_ = std::move(FFTWWrapper<Real>());
                fftHelperC_ = std::move(FFTWWrapper<Real>());
                compressionCoefficientsA_ = RealMat(numKSumTermsA_, myGridDimensionA_);
                compressionCoefficientsB_ = RealMat(numKSumTermsB_, myGridDimensionB_);
                compressionCoefficientsC_ = RealMat(numKSumTermsC_, myGridDimensionC_);
                scratchSize = (size_t)std::max(myGridDimensionA_, numKSumTermsA) *
                              std::max(myGridDimensionB_, numKSumTermsB) * std::max(myGridDimensionC_, numKSumTermsC);
            } else {
                gridDimensionA_ = findGridSize(dimA, {numNodesA_});
                gridDimensionB_ = findGridSize(dimB, {numNodesB_ * numNodesC_});
                gridDimensionC_ = findGridSize(dimC, {numNodesA_ * numNodesC_, numNodesB_ * numNodesC_});
                gridPaddingA = gridPaddingB = gridPaddingC = 0;
                myGridDimensionA_ = gridDimensionA_ / numNodesA_;
                myGridDimensionB_ = gridDimensionB_ / numNodesB_;
                myGridDimensionC_ = gridDimensionC_ / numNodesC_;
                complexGridDimensionA_ = gridDimensionA_ / 2 + 1;
                myComplexGridDimensionA_ = myGridDimensionA_ / 2 + 1;
                numKSumTermsA_ = gridDimensionA_;
                numKSumTermsB_ = gridDimensionB_;
                numKSumTermsC_ = gridDimensionC_;
                myNumKSumTermsA_ = myComplexGridDimensionA_;
                myNumKSumTermsB_ = myGridDimensionB_ / numNodesC_;
                myNumKSumTermsC_ = gridDimensionC_;
                myFirstGridPointA_ = myNodeRankA_ * myGridDimensionA_;
                myFirstGridPointB_ = myNodeRankB_ * myGridDimensionB_;
                myFirstGridPointC_ = myNodeRankC_ * myGridDimensionC_;
                firstKSumTermA_ = myNodeRankA_ * myComplexGridDimensionA_;
                firstKSumTermB_ = myNodeRankB_ * myGridDimensionB_ + myNodeRankC_ * myGridDimensionB_ / numNodesC_;
                firstKSumTermC_ = 0;
                fftHelperA_ = std::move(FFTWWrapper<Real>(gridDimensionA_));
                fftHelperB_ = std::move(FFTWWrapper<Real>(gridDimensionB_));
                fftHelperC_ = std::move(FFTWWrapper<Real>(gridDimensionC_));
                compressionCoefficientsA_ = RealMat();
                compressionCoefficientsB_ = RealMat();
                compressionCoefficientsC_ = RealMat();
                scratchSize = (size_t)myGridDimensionC_ * myComplexGridDimensionA_ * myGridDimensionB_;
            }

            // Grid iterators to correctly wrap the grid when using splines.
            gridIteratorA_ = makeGridIterator(gridDimensionA_, myFirstGridPointA_,
                                              myFirstGridPointA_ + myGridDimensionA_, gridPaddingA);
            gridIteratorB_ = makeGridIterator(gridDimensionB_, myFirstGridPointB_,
                                              myFirstGridPointB_ + myGridDimensionB_, gridPaddingB);
            gridIteratorC_ = makeGridIterator(gridDimensionC_, myFirstGridPointC_,
                                              myFirstGridPointC_ + myGridDimensionC_, gridPaddingC);

            // Divide C grid points among threads to avoid race conditions.
            threadedGridIteratorC_.clear();
            for (int thread = 0; thread < nThreads_; ++thread) {
                GridIterator myIterator;
                for (int cGridPoint = 0; cGridPoint < gridDimensionC_; ++cGridPoint) {
                    std::vector<std::pair<short, short>> splineIterator;
                    for (const auto &fullIterator : gridIteratorC_[cGridPoint]) {
                        if (fullIterator.first % nThreads_ == thread) {
                            splineIterator.push_back(fullIterator);
                        }
                    }
                    splineIterator.shrink_to_fit();
                    myIterator.push_back(splineIterator);
                }
                myIterator.shrink_to_fit();
                threadedGridIteratorC_.push_back(myIterator);
            }
            threadedGridIteratorC_.shrink_to_fit();

            // Assign a large default so that uninitialized values end up generating zeros later on
            mValsA_.resize(myNumKSumTermsA_, 99);
            mValsB_.resize(myNumKSumTermsB_, 99);
            mValsC_.resize(myNumKSumTermsC_, 99);
            if (algorithm == AlgorithmType::CompressedPME) {
                // For compressed PME we order the m values as 0, 1, -1, 2, -2, ..., Kmax, -Kmax
                // because we need to guarantee that +/- m pairs live on the same node for the virial.
                mValsA_[0] = 0;
                int startA = myNodeRankA_ ? 0 : 1;
                for (int k = startA; k < (myNumKSumTermsA_ + (numNodesA_ == 1)) / 2; ++k) {
                    int m = myNodeRankA_ * myNumKSumTermsA_ / 2 + k;
                    mValsA_[startA + 2 * (k - startA)] = m;
                    mValsA_[startA + 2 * (k - startA) + 1] = -m;
                }
                mValsB_[0] = 0;
                int startB = myNodeRankB_ ? 0 : 1;
                for (int k = startB; k < (myNumKSumTermsB_ + (numNodesB_ == 1)) / 2; ++k) {
                    int m = myNodeRankB_ * myNumKSumTermsB_ / 2 + k;
                    mValsB_[startB + 2 * (k - startB)] = m;
                    mValsB_[startB + 2 * (k - startB) + 1] = -m;
                }
                mValsC_[0] = 0;
                int startC = myNodeRankC_ ? 0 : 1;
                for (int k = startC; k < (myNumKSumTermsC_ + (numNodesC_ == 1)) / 2; ++k) {
                    int m = myNodeRankC_ * myNumKSumTermsC_ / 2 + k;
                    mValsC_[startC + 2 * (k - startC)] = m;
                    mValsC_[startC + 2 * (k - startC) + 1] = -m;
                }

                std::fill(compressionCoefficientsA_[0], compressionCoefficientsA_[1], 1);
                for (int node = 0; node < numNodesA_; ++node) {
                    int offset = node ? 0 : 1;
                    for (int m = offset; m < (myNumKSumTermsA_ + (numNodesA_ == 1)) / 2; ++m) {
                        int fullM = m + node * myNumKSumTermsA_ / 2;
                        Real *rowPtr = compressionCoefficientsA_[offset + 2 * (fullM - offset)];
                        for (int n = 0; n < myGridDimensionA_; ++n) {
                            Real exponent = 2 * HELPME_PI * fullM * (n + myFirstGridPointA_) / gridDimensionA_;
                            rowPtr[n] = std::sqrt(2) * std::cos(exponent);
                            rowPtr[n + myGridDimensionA_] = std::sqrt(2) * std::sin(exponent);
                        }
                    }
                }
                std::fill(compressionCoefficientsB_[0], compressionCoefficientsB_[1], 1);
                for (int node = 0; node < numNodesB_; ++node) {
                    int offset = node ? 0 : 1;
                    for (int m = offset; m < (myNumKSumTermsB_ + (numNodesB_ == 1)) / 2; ++m) {
                        int fullM = m + node * myNumKSumTermsB_ / 2;
                        Real *rowPtr = compressionCoefficientsB_[offset + 2 * (fullM - offset)];
                        for (int n = 0; n < myGridDimensionB_; ++n) {
                            Real exponent = 2 * HELPME_PI * fullM * (n + myFirstGridPointB_) / gridDimensionB_;
                            rowPtr[n] = std::sqrt(2) * std::cos(exponent);
                            rowPtr[n + myGridDimensionB_] = std::sqrt(2) * std::sin(exponent);
                        }
                    }
                }
                std::fill(compressionCoefficientsC_[0], compressionCoefficientsC_[1], 1);
                for (int node = 0; node < numNodesC_; ++node) {
                    int offset = node ? 0 : 1;
                    for (int m = offset; m < (myNumKSumTermsC_ + (numNodesC_ == 1)) / 2; ++m) {
                        int fullM = m + node * myNumKSumTermsC_ / 2;
                        Real *rowPtr = compressionCoefficientsC_[offset + 2 * (fullM - offset)];
                        for (int n = 0; n < myGridDimensionC_; ++n) {
                            Real exponent = 2 * HELPME_PI * fullM * (n + myFirstGridPointC_) / gridDimensionC_;
                            rowPtr[n] = std::sqrt(2) * std::cos(exponent);
                            rowPtr[n + myGridDimensionC_] = std::sqrt(2) * std::sin(exponent);
                        }
                    }
                }
                // Fourier space spline norms.
                Spline spline = Spline(0, 0, splineOrder_, 0);
                splineModA_ = spline.invSplineModuli(gridDimensionA_, mValsA_);
                splineModB_ = spline.invSplineModuli(gridDimensionB_, mValsB_);
                splineModC_ = spline.invSplineModuli(gridDimensionC_, mValsC_);
            } else {
                // For conventional PME we order the m values as 0, 1, 2, 3, .., Kmax, -Kmax, -Kmax+1, .., -2, -1
                // because this is consistent with the ordering of m values that emerge from the FFT.
                for (int ka = 0; ka < myNumKSumTermsA_; ++ka) {
                    mValsA_[ka] = firstKSumTermA_ +
                                  (ka + firstKSumTermA_ >= (gridDimensionA_ + 1) / 2 ? ka - gridDimensionA_ : ka);
                }
                for (int kb = 0; kb < myNumKSumTermsB_; ++kb) {
                    mValsB_[kb] = firstKSumTermB_ +
                                  (kb + firstKSumTermB_ >= (gridDimensionB_ + 1) / 2 ? kb - gridDimensionB_ : kb);
                }
                for (int kc = 0; kc < myNumKSumTermsC_; ++kc) {
                    mValsC_[kc] = firstKSumTermC_ +
                                  (kc + firstKSumTermC_ >= (gridDimensionC_ + 1) / 2 ? kc - gridDimensionC_ : kc);
                }
                // Fourier space spline norms.
                Spline spline = Spline(0, 0, splineOrder_, 0);
                auto fullSplineModA = spline.invSplineModuli(gridDimensionA_);
                auto fullSplineModB = spline.invSplineModuli(gridDimensionB_);
                auto fullSplineModC = spline.invSplineModuli(gridDimensionC_);

                scaledRecVecs_ = recVecs_.clone();
                scaledRecVecs_.row(0) *= gridDimensionA_;
                scaledRecVecs_.row(1) *= gridDimensionB_;
                scaledRecVecs_.row(2) *= gridDimensionC_;

                splineModA_.resize(myNumKSumTermsA_);
                splineModB_.resize(myNumKSumTermsB_);
                splineModC_.resize(myNumKSumTermsC_);
                std::copy(&fullSplineModA[firstKSumTermA_], &fullSplineModA[firstKSumTermA_ + myNumKSumTermsA_],
                          splineModA_.begin());
                std::copy(&fullSplineModB[firstKSumTermB_], &fullSplineModB[firstKSumTermB_ + myNumKSumTermsB_],
                          splineModB_.begin());
                std::copy(&fullSplineModC[firstKSumTermC_], &fullSplineModC[firstKSumTermC_ + myNumKSumTermsC_],
                          splineModC_.begin());
            }

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

            subsetOfCAlongA_ = myGridDimensionC_ / numNodesA_;
            subsetOfCAlongB_ = myGridDimensionC_ / numNodesB_;
            subsetOfBAlongC_ = myGridDimensionB_ / numNodesC_;

            workSpace1_ = helpme::vector<Complex>(scratchSize);
            workSpace2_ = helpme::vector<Complex>(scratchSize);
#if HAVE_MKL
            mkl_set_num_threads(nThreads_);
#endif
        }
    }

   public:
    /*!
     * \brief Spread the parameters onto the charge grid.  Generally this shouldn't be called;
     *        use the various computeE() methods instead. This the more efficient version that filters
     *        the atom list and uses pre-computed splines.  Therefore, the splineCache_
     *        member must have been updated via a call to filterAtomsAndBuildSplineCache() first.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
     * \return realGrid the array of discretized parameters (stored in CBA order).
     */
    Real *spreadParameters(int parameterAngMom, const RealMat &parameters) {
        Real *realGrid = reinterpret_cast<Real *>(workSpace1_.data());
        updateAngMomIterator(parameterAngMom);

        // We need to figure out whether the incoming parameters need to be transformed to scaled fractional
        // coordinates or not, which is only needed for angular momentum higher than zero.
        RealMat tempParams;
        if (parameterAngMom) {
            tempParams = cartesianTransform(parameterAngMom, false, scaledRecVecs_.transpose(), parameters);
        }
        const auto &fractionalParameters = parameterAngMom ? tempParams : parameters;

        int nComponents = nCartesian(parameterAngMom);
        size_t numBA = (size_t)myGridDimensionB_ * myGridDimensionA_;
#pragma omp parallel num_threads(nThreads_)
        {
#ifdef _OPENMP
            int threadID = omp_get_thread_num();
#else
            int threadID = 0;
#endif
            for (size_t row = threadID; row < myGridDimensionC_; row += nThreads_) {
                std::fill(&realGrid[row * numBA], &realGrid[(row + 1) * numBA], Real(0));
            }
            for (const auto &spline : splinesPerThread_[threadID]) {
                const auto &cacheEntry = splineCache_[spline];
                const int &atom = cacheEntry.absoluteAtomNumber;
                const auto &splineA = cacheEntry.aSpline;
                const auto &splineB = cacheEntry.bSpline;
                const auto &splineC = cacheEntry.cSpline;
                spreadParametersImpl(atom, realGrid, nComponents, splineA, splineB, splineC, fractionalParameters,
                                     threadID);
            }
        }
        return realGrid;
    }

    /*!
     * \brief filterAtomsAndBuildSplineCache builds a list of BSplines for only the atoms to be handled by this node.
     * \param splineDerivativeLevel the derivative level (parameter angular momentum + energy derivative level) of the
     * BSplines.
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     */
    void filterAtomsAndBuildSplineCache(int splineDerivativeLevel, const RealMat &coords) {
        assertInitialized();
        constexpr float EPS = 1e-6;

        size_t nAtoms = coords.nRows();
        numAtomsPerThread_.resize(nThreads_);
        splinesPerThread_.resize(nThreads_);
        gridAtomList_.resize(gridDimensionC_);

// Classify atoms to their worker threads first, then construct splines for each thread
#pragma omp parallel num_threads(nThreads_)
        {
#ifdef _OPENMP
            int threadID = omp_get_thread_num();
#else
            int threadID = 0;
#endif
            for (size_t row = threadID; row < gridDimensionC_; row += nThreads_) {
                gridAtomList_[row].clear();
            }
            auto &mySplineList = splinesPerThread_[threadID];
            const auto &gridIteratorC = threadedGridIteratorC_[threadID];
            mySplineList.clear();
            size_t myNumAtoms = 0;
            for (int atom = 0; atom < nAtoms; ++atom) {
                const Real *atomCoords = coords[atom];
                Real cCoord = atomCoords[0] * recVecs_(0, 2) + atomCoords[1] * recVecs_(1, 2) +
                              atomCoords[2] * recVecs_(2, 2) - EPS;
                cCoord -= floor(cCoord);
                short cStartingGridPoint = gridDimensionC_ * cCoord;
                size_t thisAtomsThread = cStartingGridPoint % nThreads_;
                const auto &cGridIterator = gridIteratorC_[cStartingGridPoint];
                if (cGridIterator.size() && thisAtomsThread == threadID) {
                    Real aCoord = atomCoords[0] * recVecs_(0, 0) + atomCoords[1] * recVecs_(1, 0) +
                                  atomCoords[2] * recVecs_(2, 0) - EPS;
                    Real bCoord = atomCoords[0] * recVecs_(0, 1) + atomCoords[1] * recVecs_(1, 1) +
                                  atomCoords[2] * recVecs_(2, 1) - EPS;
                    // Make sure the fractional coordinates fall in the range 0 <= s < 1
                    aCoord -= floor(aCoord);
                    bCoord -= floor(bCoord);
                    short aStartingGridPoint = gridDimensionA_ * aCoord;
                    short bStartingGridPoint = gridDimensionB_ * bCoord;
                    const auto &aGridIterator = gridIteratorA_[aStartingGridPoint];
                    const auto &bGridIterator = gridIteratorB_[bStartingGridPoint];
                    uint32_t startingGridPoint = cStartingGridPoint * gridDimensionB_ * gridDimensionA_ +
                                                 bStartingGridPoint * gridDimensionA_ + aStartingGridPoint;
                    if (aGridIterator.size() && bGridIterator.size()) {
                        gridAtomList_[cStartingGridPoint].emplace(startingGridPoint, atom);
                        ++myNumAtoms;
                    }
                }
            }
            numAtomsPerThread_[threadID] = myNumAtoms;
        }

        // We could intervene here and do some load balancing by inspecting the list.  Currently
        // the lazy approach of just assuming that the atoms are evenly distributed along c is used.

        size_t numCacheEntries = std::accumulate(numAtomsPerThread_.begin(), numAtomsPerThread_.end(), 0);
        // Now we know how many atoms we loop over the dense list, redefining nAtoms accordingly.
        // The first stage above is to get the number of atoms, so we can avoid calling push_back
        // and thus avoid the many memory allocations.  If the cache is too small, grow it by a
        // certain scale factor to try and minimize allocations in a not-too-wasteful manner.
        if (splineCache_.size() < numCacheEntries) {
            size_t newSize = static_cast<size_t>(1.2 * numCacheEntries);
            for (int atom = splineCache_.size(); atom < newSize; ++atom)
                splineCache_.emplace_back(splineOrder_, splineDerivativeLevel);
        }
        std::vector<size_t> threadOffset(nThreads_, 0);
        for (int thread = 1; thread < nThreads_; ++thread) {
            threadOffset[thread] = threadOffset[thread - 1] + numAtomsPerThread_[thread - 1];
        }

#pragma omp parallel num_threads(nThreads_)
        {
#ifdef _OPENMP
            int threadID = omp_get_thread_num();
#else
            int threadID = 0;
#endif
            size_t entry = threadOffset[threadID];
            for (size_t cRow = threadID; cRow < gridDimensionC_; cRow += nThreads_) {
                for (const auto &gridPointAndAtom : gridAtomList_[cRow]) {
                    size_t atom = gridPointAndAtom.second;
                    const Real *atomCoords = coords[atom];
                    Real aCoord = atomCoords[0] * recVecs_(0, 0) + atomCoords[1] * recVecs_(1, 0) +
                                  atomCoords[2] * recVecs_(2, 0) - EPS;
                    Real bCoord = atomCoords[0] * recVecs_(0, 1) + atomCoords[1] * recVecs_(1, 1) +
                                  atomCoords[2] * recVecs_(2, 1) - EPS;
                    Real cCoord = atomCoords[0] * recVecs_(0, 2) + atomCoords[1] * recVecs_(1, 2) +
                                  atomCoords[2] * recVecs_(2, 2) - EPS;
                    // Make sure the fractional coordinates fall in the range 0 <= s < 1
                    aCoord -= floor(aCoord);
                    bCoord -= floor(bCoord);
                    cCoord -= floor(cCoord);
                    short aStartingGridPoint = gridDimensionA_ * aCoord;
                    short bStartingGridPoint = gridDimensionB_ * bCoord;
                    short cStartingGridPoint = gridDimensionC_ * cCoord;
                    auto &atomSplines = splineCache_[entry++];
                    atomSplines.absoluteAtomNumber = atom;
                    atomSplines.aSpline.update(aStartingGridPoint, gridDimensionA_ * aCoord - aStartingGridPoint,
                                               splineOrder_, splineDerivativeLevel);
                    atomSplines.bSpline.update(bStartingGridPoint, gridDimensionB_ * bCoord - bStartingGridPoint,
                                               splineOrder_, splineDerivativeLevel);
                    atomSplines.cSpline.update(cStartingGridPoint, gridDimensionC_ * cCoord - cStartingGridPoint,
                                               splineOrder_, splineDerivativeLevel);
                }
            }
        }

// Finally, find all of the splines that this thread will need to handle
#pragma omp parallel num_threads(nThreads_)
        {
#ifdef _OPENMP
            int threadID = omp_get_thread_num();
#else
            int threadID = 0;
#endif
            auto &mySplineList = splinesPerThread_[threadID];
            mySplineList.clear();
            const auto &gridIteratorC = threadedGridIteratorC_[threadID];
            size_t count = 0;
            for (size_t atom = 0; atom < numCacheEntries; ++atom) {
                if (gridIteratorC[splineCache_[atom].cSpline.startingGridPoint()].size()) {
                    mySplineList.emplace_back(count);
                }
                ++count;
            }
        }
    }

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
     * \brief minimumImageDeltaR Computes deltaR = positionJ - positionI, applying the minimum image convention to the
     * result \param positionI \param positionJ \return minimum image deltaR
     */
    std::array<Real, 3> minimumImageDeltaR(const typename helpme::Matrix<Real>::sliceIterator &positionI,
                                           const typename helpme::Matrix<Real>::sliceIterator &positionJ) {
        // This implementation could be specialized for orthorhombic unit cells, but we stick with a general
        // implementation for now. The difference in real (R) space
        Real dxR = positionJ[0] - positionI[0];
        Real dyR = positionJ[1] - positionI[1];
        Real dzR = positionJ[2] - positionI[2];
        // Convert to fractional coordinate (S) space
        Real dxS = recVecs_[0][0] * dxR + recVecs_[0][1] * dyR + recVecs_[0][2] * dzR;
        Real dyS = recVecs_[1][0] * dxR + recVecs_[1][1] * dyR + recVecs_[1][2] * dzR;
        Real dzS = recVecs_[2][0] * dxR + recVecs_[2][1] * dyR + recVecs_[2][2] * dzR;
        // Apply translations in fractional coordinates to find the shift vectors
        Real sxS = std::floor(dxS + 0.5f);
        Real syS = std::floor(dyS + 0.5f);
        Real szS = std::floor(dzS + 0.5f);
        // Convert fractional coordinate shifts to real space
        Real sxR = boxVecs_[0][0] * sxS + boxVecs_[0][1] * syS + boxVecs_[0][2] * szS;
        Real syR = boxVecs_[1][0] * sxS + boxVecs_[1][1] * syS + boxVecs_[1][2] * szS;
        Real szR = boxVecs_[2][0] * sxS + boxVecs_[2][1] * syS + boxVecs_[2][2] * szS;
        // Shift the difference vector to find the minimum image
        return {dxR - sxR, dyR - syR, dzR - szR};
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
        if (A != cellA_ || B != cellB_ || C != cellC_ || alpha != cellAlpha_ || beta != cellBeta_ ||
            gamma != cellGamma_ || latticeType != latticeType_) {
            if (latticeType == LatticeType::ShapeMatrix) {
                RealMat HtH(3, 3);
                HtH(0, 0) = A * A;
                HtH(1, 1) = B * B;
                HtH(2, 2) = C * C;
                const float TOL = 1e-4f;
                // Check for angles very close to 90, to avoid noise from the eigensolver later on.
                HtH(0, 1) = HtH(1, 0) = std::abs(gamma - 90) < TOL ? 0 : A * B * std::cos(HELPME_PI * gamma / 180);
                HtH(0, 2) = HtH(2, 0) = std::abs(beta - 90) < TOL ? 0 : A * C * std::cos(HELPME_PI * beta / 180);
                HtH(1, 2) = HtH(2, 1) = std::abs(alpha - 90) < TOL ? 0 : B * C * std::cos(HELPME_PI * alpha / 180);

                auto eigenTuple = HtH.diagonalize();
                RealMat evalsReal = std::get<0>(eigenTuple);
                RealMat evecs = std::get<1>(eigenTuple);
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
                boxVecs_(1, 0) = B * std::cos(HELPME_PI / 180 * gamma);
                boxVecs_(1, 1) = B * std::sin(HELPME_PI / 180 * gamma);
                boxVecs_(1, 2) = 0;
                boxVecs_(2, 0) = C * std::cos(HELPME_PI / 180 * beta);
                boxVecs_(2, 1) =
                    (B * C * cos(HELPME_PI / 180 * alpha) - boxVecs_(2, 0) * boxVecs_(1, 0)) / boxVecs_(1, 1);
                boxVecs_(2, 2) = std::sqrt(C * C - boxVecs_(2, 0) * boxVecs_(2, 0) - boxVecs_(2, 1) * boxVecs_(2, 1));
            } else {
                throw std::runtime_error("Unknown lattice type in setLatticeVectors");
            }
            recVecs_ = boxVecs_.inverse();
            scaledRecVecs_ = recVecs_.clone();
            scaledRecVecs_.row(0) *= gridDimensionA_;
            scaledRecVecs_.row(1) *= gridDimensionB_;
            scaledRecVecs_.row(2) *= gridDimensionC_;
            cellA_ = A;
            cellB_ = B;
            cellC_ = C;
            cellAlpha_ = alpha;
            cellBeta_ = beta;
            cellGamma_ = gamma;
            latticeType_ = latticeType;
            unitCellHasChanged_ = true;
        } else {
            unitCellHasChanged_ = false;
        }
    }

    /*!
     * \brief Performs the forward 3D FFT of the discretized parameter grid using the compressed PME algorithm.
     * \param realGrid the array of discretized parameters (stored in CBA order,
     *                 with A being the fast running index) to be transformed.
     * \return Pointer to the transformed grid, which is stored in one of the buffers in BAC order.
     */
    Real *compressedForwardTransform(Real *realGrid) {
        Real *__restrict__ buffer1, *__restrict__ buffer2;
        if (realGrid == reinterpret_cast<Real *>(workSpace1_.data())) {
            buffer1 = reinterpret_cast<Real *>(workSpace2_.data());
            buffer2 = reinterpret_cast<Real *>(workSpace1_.data());
        } else {
            buffer1 = reinterpret_cast<Real *>(workSpace1_.data());
            buffer2 = reinterpret_cast<Real *>(workSpace2_.data());
        }
        // Transform A index
        contractABxCWithDxC<Real>(realGrid, compressionCoefficientsA_[0], myGridDimensionC_ * myGridDimensionB_,
                                  myGridDimensionA_, numKSumTermsA_, buffer1);
        // Sort CBA->CAB
        permuteABCtoACB(buffer1, myGridDimensionC_, myGridDimensionB_, numKSumTermsA_, buffer2, nThreads_);
        // Transform B index
        contractABxCWithDxC<Real>(buffer2, compressionCoefficientsB_[0], myGridDimensionC_ * numKSumTermsA_,
                                  myGridDimensionB_, numKSumTermsB_, buffer1);
        // Sort CAB->BAC
        permuteABCtoCBA(buffer1, myGridDimensionC_, numKSumTermsA_, numKSumTermsB_, buffer2, nThreads_);
        // Transform C index
        contractABxCWithDxC<Real>(buffer2, compressionCoefficientsC_[0], numKSumTermsB_ * numKSumTermsA_,
                                  myGridDimensionC_, numKSumTermsC_, buffer1);

#if HAVE_MPI == 1
        int numNodes = numNodesA_ * numNodesB_ * numNodesC_;
        if (numNodes > 1) {
            // Resort the data to be grouped by node, for communication
            for (int node = 0; node < numNodes; ++node) {
                int nodeStartA = myNumKSumTermsA_ * (node % numNodesA_);
                int nodeStartB = myNumKSumTermsB_ * ((node % (numNodesB_ * numNodesA_)) / numNodesA_);
                int nodeStartC = myNumKSumTermsC_ * (node / (numNodesB_ * numNodesA_));
                Real *outPtr = buffer2 + node * myNumKSumTermsA_ * myNumKSumTermsB_ * myNumKSumTermsC_;
                for (int B = 0; B < myNumKSumTermsB_; ++B) {
                    const Real *inPtrB = buffer1 + (nodeStartB + B) * numKSumTermsA_ * numKSumTermsC_;
                    for (int A = 0; A < myNumKSumTermsA_; ++A) {
                        const Real *inPtrBA = inPtrB + (nodeStartA + A) * numKSumTermsC_;
                        const Real *inPtrBAC = inPtrBA + nodeStartC;
                        std::copy(inPtrBAC, inPtrBAC + myNumKSumTermsC_, outPtr);
                        outPtr += myNumKSumTermsC_;
                    }
                }
            }
            mpiCommunicator_->reduceScatterBlock(buffer2, buffer1,
                                                 myNumKSumTermsA_ * myNumKSumTermsB_ * myNumKSumTermsC_);
        }
#endif
        return buffer1;
    }

    /*!
     * \brief Performs the forward 3D FFT of the discretized parameter grid.
     * \param realGrid the array of discretized parameters (stored in CBA order,
     *                 with A being the fast running index) to be transformed.
     * \return Pointer to the transformed grid, which is stored in one of the buffers in BAC order.
     */
    Complex *forwardTransform(Real *realGrid) {
        Real *__restrict__ realCBA;
        Complex *__restrict__ buffer1, *__restrict__ buffer2;
        if (realGrid == reinterpret_cast<Real *>(workSpace1_.data())) {
            realCBA = reinterpret_cast<Real *>(workSpace2_.data());
            buffer1 = workSpace2_.data();
            buffer2 = workSpace1_.data();
        } else {
            realCBA = reinterpret_cast<Real *>(workSpace1_.data());
            buffer1 = workSpace1_.data();
            buffer2 = workSpace2_.data();
        }

#if HAVE_MPI == 1
        if (numNodesA_ > 1) {
            // Communicate A along columns
            mpiCommunicatorA_->allToAll(realGrid, realCBA, subsetOfCAlongA_ * myGridDimensionA_ * myGridDimensionB_);
            // Resort the data to end up with realGrid holding a full row of A data, for B pencil and C subset.
            for (int c = 0; c < subsetOfCAlongA_; ++c) {
                Real *outC = realGrid + c * myGridDimensionB_ * gridDimensionA_;
                for (int b = 0; b < myGridDimensionB_; ++b) {
                    for (int chunk = 0; chunk < numNodesA_; ++chunk) {
                        Real *inPtr = realCBA + (chunk * subsetOfCAlongA_ + c) * myGridDimensionB_ * myGridDimensionA_ +
                                      b * myGridDimensionA_;
                        std::copy(inPtr, inPtr + myGridDimensionA_,
                                  outC + b * gridDimensionA_ + chunk * myGridDimensionA_);
                    }
                }
            }
        }
#endif
        // Each parallel node allocates buffers of length dimA/(2 numNodesA)+1 for A, leading to a total of
        // dimA/2 + numNodesA = complexDimA+numNodesA-1 if dimA is even
        // and
        // numNodesA (dimA-1)/2 + numNodesA = complexDimA + numNodesA/2-1 if dimA is odd
        // We just allocate the larger size here, remembering that the final padding values on the last node
        // will all be allocated to zero and will not contribute to the final answer.
        const size_t scratchRowDim = complexGridDimensionA_ + numNodesA_ - 1;
        helpme::vector<Complex> buffer(nThreads_ * scratchRowDim);

// A transform, with instant sort to CAB ordering for each local block
#pragma omp parallel num_threads(nThreads_)
        {
#ifdef _OPENMP
            int threadID = omp_get_thread_num();
#else
            int threadID = 0;
#endif
            auto scratch = &buffer[threadID * scratchRowDim];
#pragma omp for
            for (int c = 0; c < subsetOfCAlongA_; ++c) {
                for (int b = 0; b < myGridDimensionB_; ++b) {
                    Real *gridPtr = realGrid + c * myGridDimensionB_ * gridDimensionA_ + b * gridDimensionA_;
                    fftHelperA_.transform(gridPtr, scratch);
                    for (int chunk = 0; chunk < numNodesA_; ++chunk) {
                        for (int a = 0; a < myComplexGridDimensionA_; ++a) {
                            buffer1[(chunk * subsetOfCAlongA_ + c) * myComplexGridDimensionA_ * myGridDimensionB_ +
                                    a * myGridDimensionB_ + b] = scratch[chunk * myComplexGridDimensionA_ + a];
                        }
                    }
                }
            }
        }

#if HAVE_MPI == 1
        // Communicate A back to blocks
        if (numNodesA_ > 1) {
            mpiCommunicatorA_->allToAll(buffer1, buffer2,
                                        subsetOfCAlongA_ * myComplexGridDimensionA_ * myGridDimensionB_);
            std::swap(buffer1, buffer2);
        }

        // Communicate B along rows
        if (numNodesB_ > 1) {
            mpiCommunicatorB_->allToAll(buffer1, buffer2,
                                        subsetOfCAlongB_ * myComplexGridDimensionA_ * myGridDimensionB_);
            // Resort the data to end up with the buffer holding a full row of B data, for A pencil and C subset.
            for (int c = 0; c < subsetOfCAlongB_; ++c) {
                Complex *cPtr = buffer1 + c * myComplexGridDimensionA_ * gridDimensionB_;
                for (int a = 0; a < myComplexGridDimensionA_; ++a) {
                    for (int chunk = 0; chunk < numNodesB_; ++chunk) {
                        Complex *inPtr = buffer2 +
                                         (chunk * subsetOfCAlongB_ + c) * myComplexGridDimensionA_ * myGridDimensionB_ +
                                         a * myGridDimensionB_;
                        std::copy(inPtr, inPtr + myGridDimensionB_,
                                  cPtr + a * gridDimensionB_ + chunk * myGridDimensionB_);
                    }
                }
            }
        }
#endif

        // B transform
        size_t numCA = (size_t)subsetOfCAlongB_ * myComplexGridDimensionA_;
#pragma omp parallel for num_threads(nThreads_)
        for (size_t ca = 0; ca < numCA; ++ca) {
            fftHelperB_.transform(buffer1 + ca * gridDimensionB_, FFTW_FORWARD);
        }

#if HAVE_MPI == 1
        if (numNodesB_ > 1) {
            for (int c = 0; c < subsetOfCAlongB_; ++c) {
                Complex *zPtr = buffer1 + c * myComplexGridDimensionA_ * gridDimensionB_;
                for (int a = 0; a < myComplexGridDimensionA_; ++a) {
                    for (int chunk = 0; chunk < numNodesB_; ++chunk) {
                        Complex *inPtr = zPtr + a * gridDimensionB_ + chunk * myGridDimensionB_;
                        Complex *outPtr =
                            buffer2 + (chunk * subsetOfCAlongB_ + c) * myComplexGridDimensionA_ * myGridDimensionB_ +
                            a * myGridDimensionB_;
                        std::copy(inPtr, inPtr + myGridDimensionB_, outPtr);
                    }
                }
            }
            // Communicate B back to blocks
            mpiCommunicatorB_->allToAll(buffer2, buffer1,
                                        subsetOfCAlongB_ * myComplexGridDimensionA_ * myGridDimensionB_);
        }
#endif
        // sort local blocks from CAB to BAC order
        permuteABCtoCBA(buffer1, myGridDimensionC_, myComplexGridDimensionA_, myGridDimensionB_, buffer2, nThreads_);

#if HAVE_MPI == 1
        if (numNodesC_ > 1) {
            // Communicate C along columns
            mpiCommunicatorC_->allToAll(buffer2, buffer1,
                                        subsetOfBAlongC_ * myComplexGridDimensionA_ * myGridDimensionC_);
            for (int b = 0; b < subsetOfBAlongC_; ++b) {
                Complex *outPtrB = buffer2 + b * myComplexGridDimensionA_ * gridDimensionC_;
                for (int a = 0; a < myComplexGridDimensionA_; ++a) {
                    Complex *outPtrBA = outPtrB + a * gridDimensionC_;
                    for (int chunk = 0; chunk < numNodesC_; ++chunk) {
                        Complex *inPtr = buffer1 +
                                         (chunk * subsetOfBAlongC_ + b) * myComplexGridDimensionA_ * myGridDimensionC_ +
                                         a * myGridDimensionC_;
                        std::copy(inPtr, inPtr + myGridDimensionC_, outPtrBA + chunk * myGridDimensionC_);
                    }
                }
            }
        }
#endif
        // C transform
        size_t numBA = (size_t)subsetOfBAlongC_ * myComplexGridDimensionA_;
#pragma omp parallel for num_threads(nThreads_)
        for (size_t ba = 0; ba < numBA; ++ba) {
            fftHelperC_.transform(buffer2 + ba * gridDimensionC_, FFTW_FORWARD);
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
        Complex *__restrict__ buffer1, *__restrict__ buffer2;
        // Setup scratch, taking care not to overwrite the convolved grid.
        if (convolvedGrid == workSpace1_.data()) {
            buffer1 = workSpace2_.data();
            buffer2 = workSpace1_.data();
        } else {
            buffer1 = workSpace1_.data();
            buffer2 = workSpace2_.data();
        }

        // C transform
        size_t numYX = (size_t)subsetOfBAlongC_ * myComplexGridDimensionA_;
#pragma omp parallel for num_threads(nThreads_)
        for (size_t yx = 0; yx < numYX; ++yx) {
            fftHelperC_.transform(convolvedGrid + yx * gridDimensionC_, FFTW_BACKWARD);
        }

#if HAVE_MPI == 1
        if (numNodesC_ > 1) {
            // Communicate C back to blocks
            for (int b = 0; b < subsetOfBAlongC_; ++b) {
                Complex *inPtrB = convolvedGrid + b * myComplexGridDimensionA_ * gridDimensionC_;
                for (int a = 0; a < myComplexGridDimensionA_; ++a) {
                    Complex *inPtrBA = inPtrB + a * gridDimensionC_;
                    for (int chunk = 0; chunk < numNodesC_; ++chunk) {
                        Complex *inPtrBAC = inPtrBA + chunk * myGridDimensionC_;
                        Complex *outPtr =
                            buffer1 + (chunk * subsetOfBAlongC_ + b) * myComplexGridDimensionA_ * myGridDimensionC_ +
                            a * myGridDimensionC_;
                        std::copy(inPtrBAC, inPtrBAC + myGridDimensionC_, outPtr);
                    }
                }
            }
            mpiCommunicatorC_->allToAll(buffer1, buffer2,
                                        subsetOfBAlongC_ * myComplexGridDimensionA_ * myGridDimensionC_);
        }
#endif

        // sort local blocks from BAC to CAB order
        permuteABCtoCBA(buffer2, myGridDimensionB_, myComplexGridDimensionA_, myGridDimensionC_, buffer1, nThreads_);

#if HAVE_MPI == 1
        // Communicate B along rows
        if (numNodesB_ > 1) {
            mpiCommunicatorB_->allToAll(buffer1, buffer2,
                                        subsetOfCAlongB_ * myComplexGridDimensionA_ * myGridDimensionB_);
            // Resort the data to end up with the buffer holding a full row of B data, for A pencil and C subset.
            for (int c = 0; c < subsetOfCAlongB_; ++c) {
                Complex *cPtr = buffer1 + c * myComplexGridDimensionA_ * gridDimensionB_;
                for (int a = 0; a < myComplexGridDimensionA_; ++a) {
                    for (int chunk = 0; chunk < numNodesB_; ++chunk) {
                        Complex *inPtr = buffer2 +
                                         (chunk * subsetOfCAlongB_ + c) * myComplexGridDimensionA_ * myGridDimensionB_ +
                                         a * myGridDimensionB_;
                        std::copy(inPtr, inPtr + myGridDimensionB_,
                                  cPtr + a * gridDimensionB_ + chunk * myGridDimensionB_);
                    }
                }
            }
        }
#endif

        // B transform with instant sort of local blocks from CAB -> CBA order
        size_t numCA = (size_t)subsetOfCAlongB_ * myComplexGridDimensionA_;
#pragma omp parallel for num_threads(nThreads_)
        for (size_t ca = 0; ca < numCA; ++ca) {
            fftHelperB_.transform(buffer1 + ca * gridDimensionB_, FFTW_BACKWARD);
        }
#pragma omp parallel for num_threads(nThreads_)
        for (int c = 0; c < subsetOfCAlongB_; ++c) {
            for (int a = 0; a < myComplexGridDimensionA_; ++a) {
                int cx = c * myComplexGridDimensionA_ * gridDimensionB_ + a * gridDimensionB_;
                for (int b = 0; b < myGridDimensionB_; ++b) {
                    for (int chunk = 0; chunk < numNodesB_; ++chunk) {
                        int cb = (chunk * subsetOfCAlongB_ + c) * myGridDimensionB_ * myComplexGridDimensionA_ +
                                 b * myComplexGridDimensionA_;
                        buffer2[cb + a] = buffer1[cx + chunk * myGridDimensionB_ + b];
                    }
                }
            }
        }

#if HAVE_MPI == 1
        // Communicate B back to blocks
        if (numNodesB_ > 1) {
            mpiCommunicatorB_->allToAll(buffer2, buffer1,
                                        subsetOfCAlongB_ * myComplexGridDimensionA_ * myGridDimensionB_);
        } else {
            std::swap(buffer1, buffer2);
        }

        // Communicate A along rows
        if (numNodesA_ > 1) {
            mpiCommunicatorA_->allToAll(buffer1, buffer2,
                                        subsetOfCAlongA_ * myComplexGridDimensionA_ * myGridDimensionB_);
            // Resort the data to end up with the buffer holding a full row of A data, for B pencil and C subset.
            for (int c = 0; c < subsetOfCAlongA_; ++c) {
                Complex *cPtr = buffer1 + c * myGridDimensionB_ * complexGridDimensionA_;
                for (int b = 0; b < myGridDimensionB_; ++b) {
                    for (int chunk = 0; chunk < numNodesA_; ++chunk) {
                        Complex *inPtr = buffer2 +
                                         (chunk * subsetOfCAlongA_ + c) * myComplexGridDimensionA_ * myGridDimensionB_ +
                                         b * myComplexGridDimensionA_;
                        std::copy(inPtr, inPtr + myComplexGridDimensionA_,
                                  cPtr + b * complexGridDimensionA_ + chunk * myComplexGridDimensionA_);
                    }
                }
            }
        }
#else
        std::swap(buffer1, buffer2);
#endif

        // A transform
        Real *realGrid = reinterpret_cast<Real *>(buffer2);
#pragma omp parallel for num_threads(nThreads_)
        for (int cb = 0; cb < subsetOfCAlongA_ * myGridDimensionB_; ++cb) {
            fftHelperA_.transform(buffer1 + cb * complexGridDimensionA_, realGrid + cb * gridDimensionA_);
        }

#if HAVE_MPI == 1
        // Communicate A back to blocks
        if (numNodesA_ > 1) {
            Real *realGrid2 = reinterpret_cast<Real *>(buffer1);
            for (int c = 0; c < subsetOfCAlongA_; ++c) {
                Real *cPtr = realGrid + c * myGridDimensionB_ * gridDimensionA_;
                for (int b = 0; b < myGridDimensionB_; ++b) {
                    for (int chunk = 0; chunk < numNodesA_; ++chunk) {
                        Real *outPtr = realGrid2 +
                                       (chunk * subsetOfCAlongA_ + c) * myGridDimensionB_ * myGridDimensionA_ +
                                       b * myGridDimensionA_;
                        Real *inPtr = cPtr + b * gridDimensionA_ + chunk * myGridDimensionA_;
                        std::copy(inPtr, inPtr + myGridDimensionA_, outPtr);
                    }
                }
            }
            mpiCommunicatorA_->allToAll(realGrid2, realGrid, subsetOfCAlongA_ * myGridDimensionB_ * myGridDimensionA_);
        }
#endif
        return realGrid;
    }

    /*!
     * \brief Performs the backward 3D FFT of the discretized parameter grid using the compressed PME algorithm.
     * \param reciprocalGrid the reciprocal space potential grid (stored in BAC order,
     *                 with C being the fast running index) to be transformed.
     * \return Pointer to the transformed grid, which is stored in one of the buffers in CBA order.
     */
    Real *compressedInverseTransform(Real *reciprocalGrid) {
        Real *__restrict__ buffer1, *__restrict__ buffer2;
        if (reciprocalGrid == reinterpret_cast<Real *>(workSpace1_.data())) {
            buffer1 = reinterpret_cast<Real *>(workSpace2_.data());
            buffer2 = reinterpret_cast<Real *>(workSpace1_.data());
        } else {
            buffer1 = reinterpret_cast<Real *>(workSpace1_.data());
            buffer2 = reinterpret_cast<Real *>(workSpace2_.data());
        }
        // Make the reciprocal dimensions the fast running indices
        compressionCoefficientsA_.transposeInPlace();
        compressionCoefficientsB_.transposeInPlace();
        compressionCoefficientsC_.transposeInPlace();

#if HAVE_MPI == 1
        int numNodes = numNodesA_ * numNodesB_ * numNodesC_;
        if (numNodes > 1) {
            mpiCommunicator_->allGather(buffer2, buffer1, myNumKSumTermsA_ * myNumKSumTermsB_ * myNumKSumTermsC_);
            // Resort the data to be grouped by node, for communication
            for (int node = 0; node < numNodes; ++node) {
                int nodeStartA = myNumKSumTermsA_ * (node % numNodesA_);
                int nodeStartB = myNumKSumTermsB_ * ((node % (numNodesB_ * numNodesA_)) / numNodesA_);
                int nodeStartC = myNumKSumTermsC_ * (node / (numNodesB_ * numNodesA_));
                Real *inPtr = buffer1 + node * myNumKSumTermsA_ * myNumKSumTermsB_ * myNumKSumTermsC_;
                for (int B = 0; B < myNumKSumTermsB_; ++B) {
                    Real *outPtrB = buffer2 + (nodeStartB + B) * numKSumTermsA_ * numKSumTermsC_;
                    for (int A = 0; A < myNumKSumTermsA_; ++A) {
                        Real *outPtrBA = outPtrB + (nodeStartA + A) * numKSumTermsC_;
                        Real *outPtrBAC = outPtrBA + nodeStartC;
                        std::copy(inPtr, inPtr + myNumKSumTermsC_, outPtrBAC);
                        inPtr += myNumKSumTermsC_;
                    }
                }
            }
        }
#endif

        // Transform C index
        contractABxCWithDxC<Real>(buffer2, compressionCoefficientsC_[0], numKSumTermsB_ * numKSumTermsA_,
                                  numKSumTermsC_, myGridDimensionC_, buffer1);
        // Sort BAC->CAB
        permuteABCtoCBA(buffer1, numKSumTermsB_, numKSumTermsA_, myGridDimensionC_, buffer2, nThreads_);
        // Transform B index
        contractABxCWithDxC<Real>(buffer2, compressionCoefficientsB_[0], myGridDimensionC_ * numKSumTermsA_,
                                  numKSumTermsB_, myGridDimensionB_, buffer1);
        // Sort CAB->CBA
        permuteABCtoACB(buffer1, myGridDimensionC_, numKSumTermsA_, myGridDimensionB_, buffer2, nThreads_);
        // Transform A index
        contractABxCWithDxC<Real>(buffer2, compressionCoefficientsA_[0], myGridDimensionC_ * myGridDimensionB_,
                                  numKSumTermsA_, myGridDimensionA_, buffer1);

        // Make the grid dimensions the fast running indices again
        compressionCoefficientsA_.transposeInPlace();
        compressionCoefficientsB_.transposeInPlace();
        compressionCoefficientsC_.transposeInPlace();

        return buffer1;
    }

    /*!
     * \brief convolveE performs the convolution on a compressed PME transformed Grid
     * \param transformedGrid the pointer to the complex array holding the transformed grid in YXZ ordering.
     * \return the reciprocal space energy.
     */
    Real convolveE(Real *transformedGrid) {
        updateInfluenceFunction();
        size_t nxz = (size_t)myNumKSumTermsA_ * myNumKSumTermsC_;
        size_t nyxz = myNumKSumTermsB_ * nxz;
        bool iAmNodeZero = (myNodeRankA_ == 0 && myNodeRankB_ == 0 && myNodeRankC_ == 0);
        Real *influenceFunction = cachedInfluenceFunction_.data();
        Real energy = 0;
        if (rPower_ > 3 && iAmNodeZero) {
            // Kernels with rPower>3 are absolutely convergent and should have the m=0 term present.
            // To compute it we need sum_ij c(i)c(j), which can be obtained from the structure factor norm.
            Real prefac = 2 * scaleFactor_ * HELPME_PI * HELPME_SQRTPI * pow(kappa_, rPower_ - 3) /
                          ((rPower_ - 3) * nonTemplateGammaComputer<Real>(rPower_) * cellVolume());
            energy += prefac * transformedGrid[0] * transformedGrid[0];
        }
        if (iAmNodeZero) transformedGrid[0] = 0;
// Writing the three nested loops in one allows for better load balancing in parallel.
#pragma omp parallel for reduction(+ : energy) num_threads(nThreads_)
        for (size_t yxz = 0; yxz < nyxz; ++yxz) {
            energy += transformedGrid[yxz] * transformedGrid[yxz] * influenceFunction[yxz];
            transformedGrid[yxz] *= influenceFunction[yxz];
        }
        return energy / 2;
    }

    /*!
     * \brief convolveE performs the convolution of a standard PME transformed grid
     * \param transformedGrid the pointer to the complex array holding the transformed grid in YXZ ordering.
     * \return the reciprocal space energy.
     */
    Real convolveE(Complex *transformedGrid) {
        updateInfluenceFunction();
        size_t nxz = (size_t)myNumKSumTermsA_ * myNumKSumTermsC_;
        size_t nyxz = myNumKSumTermsB_ * nxz;
        bool iAmNodeZero = (myNodeRankA_ == 0 && myNodeRankB_ == 0 && myNodeRankC_ == 0);
        Real *influenceFunction = cachedInfluenceFunction_.data();
        bool useConjugateSymmetry = algorithmType_ == AlgorithmType::PME;

        Real energy = 0;
        if (rPower_ > 3 && iAmNodeZero) {
            // Kernels with rPower>3 are absolutely convergent and should have the m=0 term present.
            // To compute it we need sum_ij c(i)c(j), which can be obtained from the structure factor norm.
            Real prefac = 2 * scaleFactor_ * HELPME_PI * HELPME_SQRTPI * pow(kappa_, rPower_ - 3) /
                          ((rPower_ - 3) * nonTemplateGammaComputer<Real>(rPower_) * cellVolume());
            energy += prefac * std::norm(transformedGrid[0]);
        }
        if (iAmNodeZero) transformedGrid[0] = Complex(0, 0);
        const size_t numCTerms(myNumKSumTermsC_);
#pragma omp parallel for reduction(+ : energy) num_threads(nThreads_)
        for (size_t yxz = 0; yxz < nyxz; ++yxz) {
            size_t xz = yxz % nxz;
            int kx = firstKSumTermA_ + xz / numCTerms;
            // We only loop over the first nx/2+1 x values; this
            // accounts for the "missing" complex conjugate values.
            Real permPrefac = useConjugateSymmetry && kx != 0 && kx != complexGridDimensionA_ - 1 ? 2 : 1;
            Real structFactorNorm = transformedGrid[yxz].real() * transformedGrid[yxz].real() +
                                    transformedGrid[yxz].imag() * transformedGrid[yxz].imag();
            energy += permPrefac * structFactorNorm * influenceFunction[yxz];
            transformedGrid[yxz] *= influenceFunction[yxz];
        }
        return energy / 2;
    }

    /*!
     * \brief convolveEV A wrapper to determine the correct convolution function to call, including virial, for
     *        the compressed PME algorithm.
     * \param transformedGrid the pointer to the Fourier space array holding the transformed grid in YXZ ordering.
     * \param convolvedGrid the (output) pointer to the Fourier space array holding the convolved grid in YXZ ordering.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \return the reciprocal space energy.
     */
    Real convolveEV(const Real *transformedGrid, Real *&convolvedGrid, RealMat &virial) {
        convolvedGrid = transformedGrid == reinterpret_cast<Real *>(workSpace1_.data())
                            ? reinterpret_cast<Real *>(workSpace2_.data())
                            : reinterpret_cast<Real *>(workSpace1_.data());
        return convolveEVCompressedFxn_(
            myNumKSumTermsA_, myNumKSumTermsB_, myNumKSumTermsC_, firstKSumTermA_, firstKSumTermB_, firstKSumTermC_,
            scaleFactor_, transformedGrid, convolvedGrid, recVecs_, cellVolume(), kappa_, &splineModA_[0],
            &splineModB_[0], &splineModC_[0], mValsA_.data(), mValsB_.data(), mValsC_.data(), virial, nThreads_);
    }

    /*!
     * \brief convolveEV A wrapper to determine the correct convolution function to call, including virial, for
     *        the conventional PME algorithm.
     * \param transformedGrid the pointer to the complex array holding the transformed grid in YXZ ordering.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \return the reciprocal space energy.
     */
    Real convolveEV(Complex *transformedGrid, RealMat &virial) {
        return convolveEVFxn_(true, complexGridDimensionA_, myNumKSumTermsA_, myNumKSumTermsB_, myNumKSumTermsC_,
                              firstKSumTermA_, firstKSumTermB_, firstKSumTermC_, scaleFactor_, transformedGrid,
                              recVecs_, cellVolume(), kappa_, &splineModA_[0], &splineModB_[0], &splineModC_[0],
                              mValsA_.data(), mValsB_.data(), mValsC_.data(), virial, nThreads_);
    }

    /*!
     * \brief Probes the potential grid to get the forces.  Generally this shouldn't be called;
     *        use the various computeE() methods instead.  This is the faster version that uses
     *        the filtered atom list and uses pre-computed splines.  Therefore, the splineCache_
     *        member must have been updated via a call to filterAtomsAndBuildSplineCache() first.
     *
     * \param potentialGrid pointer to the array containing the potential, in ZYX order.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     * \param virial pointer to the virial vector if needed
     */
    void probeGrid(const Real *potentialGrid, int parameterAngMom, const RealMat &parameters, RealMat &forces,
                   Real *virial = nullptr) {
        updateAngMomIterator(parameterAngMom + 1);
        int nComponents = nCartesian(parameterAngMom);
        int nForceComponents = nCartesian(parameterAngMom + 1);
        const Real *paramPtr = parameters[0];
        // Find how many multiples of the cache line size are needed
        // to ensure that each thread hits a unique page.
        size_t nAtoms = std::accumulate(numAtomsPerThread_.begin(), numAtomsPerThread_.end(), 0);
        size_t rowSize = std::ceil(nForceComponents / cacheLineSizeInReals_) * cacheLineSizeInReals_;
        if (fractionalPhis_.nRows() < nAtoms || fractionalPhis_.nCols() < rowSize) {
            fractionalPhis_ = RealMat(nAtoms, rowSize);
        }

        RealMat fractionalParams;
        Real cartPhi[3];
        if (parameterAngMom) {
            fractionalParams = cartesianTransform(parameterAngMom, false, scaledRecVecs_.transpose(), parameters);
            if (virial) {
                if (parameterAngMom > 1) {
                    // The structure factor derivatives below are only implemented up to dipoles for now
                    throw std::runtime_error("Only multipoles up to L=1 are supported if the virial is requested");
                }
            }
        }
#pragma omp parallel num_threads(nThreads_)
        {
#ifdef _OPENMP
            int threadID = omp_get_thread_num();
#else
            int threadID = 0;
#endif
#pragma omp for
            for (size_t atom = 0; atom < nAtoms; ++atom) {
                const auto &cacheEntry = splineCache_[atom];
                const auto &absAtom = cacheEntry.absoluteAtomNumber;
                const auto &splineA = cacheEntry.aSpline;
                const auto &splineB = cacheEntry.bSpline;
                const auto &splineC = cacheEntry.cSpline;
                if (parameterAngMom) {
                    Real *myScratch = fractionalPhis_[threadID % nThreads_];
                    probeGridImpl(absAtom, potentialGrid, nComponents, nForceComponents, splineA, splineB, splineC,
                                  myScratch, fractionalParams[absAtom], forces[absAtom]);
                    // Add extra virial terms coming from the derivative of the structure factor.
                    // See eq. 2.16 of https://doi.org/10.1063/1.1630791 for details
                    if (virial) {
                        // Get the potential in the Cartesian basis
                        matrixVectorProduct(scaledRecVecs_, &myScratch[1], &cartPhi[0]);
                        const Real *parm = parameters[absAtom];
                        virial[0] += cartPhi[0] * parm[1];
                        virial[1] += 0.5f * (cartPhi[0] * parm[2] + cartPhi[1] * parm[1]);
                        virial[2] += cartPhi[1] * parm[2];
                        virial[3] += 0.5f * (cartPhi[0] * parm[3] + cartPhi[2] * parm[1]);
                        virial[4] += 0.5f * (cartPhi[1] * parm[3] + cartPhi[2] * parm[2]);
                        virial[5] += cartPhi[2] * parm[3];
                    }
                } else {
                    probeGridImpl(potentialGrid, splineA, splineB, splineC, paramPtr[absAtom], forces[absAtom]);
                }
            }
        }
    }

    /*!
     * \brief computeESlf computes the Ewald self interaction energy.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
        assertInitialized();
        auto prefac = slfEFxn_(parameterAngMom, kappa_, scaleFactor_);
        size_t nAtoms = parameters.nRows();
        Real sumCoefs = 0;
        for (size_t atom = 0; atom < nAtoms; ++atom) {
            sumCoefs += parameters(atom, 0) * parameters(atom, 0);
        }
        return prefac * sumCoefs;
    }

    /*!
     * \brief computeEDir computes the direct space energy.  This is provided mostly for debugging and testing
     * purposes; generally the host program should provide the pairwise interactions. \param pairList dense list of
     * atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN. \param parameterAngMom the angular momentum of
     * the parameters (0 for charges, C6 coefficients, 2 for quadrupoles, etc.). \param parameters the list of
     * parameters associated with each atom (charges, C6 coefficients, multipoles, etc...). For a parameter with
     * angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the
     * fast running index nL has the ordering
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
        if (parameterAngMom) throw std::runtime_error("Multipole direct terms have not been coded yet.");
        sanityChecks(parameterAngMom, parameters, coordinates);

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
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
        sanityChecks(parameterAngMom, parameters, coordinates);

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
            Real f = -prefactor * fKernel;
            auto force = deltaR.row(0);
            force *= f;
            forces.row(i) -= force;
            forces.row(j) += force;
        }
        return energy;
    }

    /*!
     * \brief computeEFVDir computes the direct space energy, force and virial.  This is provided mostly for
     * debugging and testing purposes; generally the host program should provide the pairwise interactions. \param
     * pairList dense list of atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN. \param parameterAngMom
     * the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles, etc.). \param
     * parameters the list of parameters associated with each atom (charges, C6 coefficients, multipoles, etc...).
     * For a parameter with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL =
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
        sanityChecks(parameterAngMom, parameters, coordinates);

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
            Real f = -prefactor * fKernel;
            RealMat dRCopy = deltaR.clone();
            auto force = dRCopy.row(0);
            force *= f;
            forces.row(i) -= force;
            forces.row(j) += force;
            virial[0][0] += force[0] * deltaR[0][0];
            virial[0][1] += 0.5f * (force[0] * deltaR[0][1] + force[1] * deltaR[0][0]);
            virial[0][2] += force[1] * deltaR[0][1];
            virial[0][3] += 0.5f * (force[0] * deltaR[0][2] + force[2] * deltaR[0][0]);
            virial[0][4] += 0.5f * (force[1] * deltaR[0][2] + force[2] * deltaR[0][1]);
            virial[0][5] += force[2] * deltaR[0][2];
        }
        return energy;
    }

    /*!
     * \brief computeEAdj computes the adjusted real space energy which extracts the energy for excluded pairs that
     * is present in reciprocal space. This is provided mostly for debugging and testing purposes; generally the
     *        host program should provide the pairwise interactions.
     * \param pairList dense list of atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
        sanityChecks(parameterAngMom, parameters, coordinates);

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
     * \brief computeEFAdj computes the adjusted energy and force.  This is provided mostly for debugging and
     * testing purposes; generally the host program should provide the pairwise interactions. \param pairList dense
     * list of atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN. \param parameterAngMom the angular
     * momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles, etc.). \param parameters the
     * list of parameters associated with each atom (charges, C6 coefficients, multipoles, etc...). For a parameter
     * with angular momentum L, a matrix of dimension nAtoms x nL is expected, where nL = (L+1)*(L+2)*(L+3)/6 and
     * the fast running index nL has the ordering
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
        sanityChecks(parameterAngMom, parameters, coordinates);

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
            Real f = -prefactor * fKernel;
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
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
        sanityChecks(parameterAngMom, parameters, coordinates);

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
            Real f = -prefactor * fKernel;
            RealMat dRCopy = deltaR.clone();
            auto force = dRCopy.row(0);
            force *= f;
            forces.row(i) -= force;
            forces.row(j) += force;
            virial[0][0] += force[0] * deltaR[0][0];
            virial[0][1] += 0.5f * (force[0] * deltaR[0][1] + force[1] * deltaR[0][0]);
            virial[0][2] += force[1] * deltaR[0][1];
            virial[0][3] += 0.5f * (force[0] * deltaR[0][2] + force[2] * deltaR[0][0]);
            virial[0][4] += 0.5f * (force[1] * deltaR[0][2] + force[2] * deltaR[0][1]);
            virial[0][5] += force[2] * deltaR[0][2];
        }
        return energy;
    }

    /*!
     * \brief Computes the full electrostatic potential at atomic sites due to point charges located at those same
     * sites. The site located at each probe location is neglected, to avoid the resulting singularity.
     * \param charges * the list of point charges (in e) associated with each particle.
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param potential the array holding the potential.  This is * a matrix of dimensions  nAtoms x 1.
     * \param sphericalCutoff the cutoff (in A) applied to the real space summations,
     * which must be no more than half of the box dimensions.
     */
    void computePAtAtomicSites(const RealMat &charges, const RealMat &coordinates, RealMat &potential,
                               Real sphericalCutoff) {
        sanityChecks(0, charges, coordinates);
        // The minumum image convention requires that the cutoff be less than half the minumum box width
        checkMinimumImageCutoff(sphericalCutoff);
        size_t nAtoms = coordinates.nRows();

        // Direct space, using simple O(N^2) algorithm.  This can be improved using a nonbonded list if needed.
        Real cutoffSquared = sphericalCutoff * sphericalCutoff;
        Real kappaSquared = kappa_ * kappa_;
#pragma omp parallel for num_threads(nThreads_)
        for (size_t i = 0; i < nAtoms; ++i) {
            const auto &coordsI = coordinates.row(i);
            Real *phiPtr = potential[i];
            for (size_t j = 0; j < nAtoms; ++j) {
                // No self interactions are included, to remove the singularity
                if (i == j) continue;
                Real qJ = charges[j][0];
                const auto &coordsJ = coordinates.row(j);
                auto RIJ = minimumImageDeltaR(coordsI, coordsJ);
                Real rSquared = RIJ[0] * RIJ[0] + RIJ[1] * RIJ[1] + RIJ[2] * RIJ[2];
                if (rSquared < cutoffSquared) {
                    *phiPtr += scaleFactor_ * qJ * dirEFxn_(rSquared, kappaSquared);
                }
            }
        }

        // Reciprocal space term
        filterAtomsAndBuildSplineCache(0, coordinates);
        auto realGrid = spreadParameters(0, charges);
        Real *potentialGrid;
        if (algorithmType_ == AlgorithmType::PME) {
            auto gridAddress = forwardTransform(realGrid);
            convolveE(gridAddress);
            potentialGrid = inverseTransform(gridAddress);
        } else if (algorithmType_ == AlgorithmType::CompressedPME) {
            auto gridAddress = compressedForwardTransform(realGrid);
            convolveE(gridAddress);
            potentialGrid = compressedInverseTransform(gridAddress);
        } else {
            std::logic_error("Unknown algorithm in helpme::computePAtAtomicSites");
        }
#pragma omp parallel for num_threads(nThreads_)
        for (size_t atom = 0; atom < nAtoms; ++atom) {
            const auto &cacheEntry = splineCache_[atom];
            const auto &absAtom = cacheEntry.absoluteAtomNumber;
            probeGridImpl(potentialGrid, 1, cacheEntry.aSpline, cacheEntry.bSpline, cacheEntry.cSpline,
                          potential[absAtom]);
        }

        // Self term - back out the contribution from the atoms at each probe site
        Real prefac = slfEFxn_(0, kappa_, scaleFactor_);
        for (size_t atom = 0; atom < nAtoms; ++atom) {
            potential[atom][0] += 2 * prefac * charges[atom][0];
        }
    }

    /*
     * \brief Runs a PME reciprocal space calculation, computing the potential and, optionally, its derivatives as
     *        well as the volume dependent part of the virial that comes from the structure factor.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.).  A negative value indicates that only the shell with |parameterAngMom| is to be considered,
     * e.g. a value of -2 specifies that only quadrupoles (and not dipoles or charges) will be provided; the input
     * matrix should have dimensions corresponding only to the number of terms in this shell.
     * \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
     * \param gridPoints the list of grid points at which the potential is needed; can be the same as the
     * coordinates.
     * \param derivativeLevel the order of the potential derivatives required; 0 is the potential, 1 is
     * (minus) the field, etc.  A negative value indicates that only the derivative with order |parameterAngMom|
     * is to be generated, e.g. -2 specifies that only the second derivative (not the potential or its gradient)
     * will be returned as output.  The output matrix should have space for only these terms, accordingly.
     * \param potential the array holding the potential.  This is a matrix of dimensions
     * nAtoms x nD, where nD is the derivative level requested.  See the details fo the parameters argument for
     * information about ordering of derivative components. N.B. this array is incremented with the potential, not
     * assigned, so take care to zero it first if only the current results are desired.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     */

    void computePVRec(int parameterAngMom, const RealMat &parameters, const RealMat &coordinates,
                      const RealMat &gridPoints, int derivativeLevel, RealMat &potential, RealMat &virial) {
        computePRecHelper(parameterAngMom, parameters, coordinates, gridPoints, derivativeLevel, potential, virial);
    }

    /*!
     * \brief Runs a PME reciprocal space calculation, computing the potential and, optionally, its derivatives.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.).  A negative value indicates that only the shell with |parameterAngMom| is to be considered,
     * e.g. a value of -2 specifies that only quadrupoles (and not dipoles or charges) will be provided; the input
     * matrix should have dimensions corresponding only to the number of terms in this shell.
     * \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
     * \param gridPoints the list of grid points at which the potential is needed; can be the same as the
     * coordinates.
     * \param derivativeLevel the order of the potential derivatives required; 0 is the potential, 1 is
     * (minus) the field, etc.  A negative value indicates that only the derivative with order |parameterAngMom|
     * is to be generated, e.g. -2 specifies that only the second derivative (not the potential or its gradient)
     * will be returned as output.  The output matrix should have space for only these terms, accordingly.
     * \param potential the array holding the potential.  This is a matrix of dimensions
     * nAtoms x nD, where nD is the derivative level requested.  See the details fo the parameters argument for
     * information about ordering of derivative components. N.B. this array is incremented with the potential, not
     * assigned, so take care to zero it first if only the current results are desired.
     */
    void computePRec(int parameterAngMom, const RealMat &parameters, const RealMat &coordinates,
                     const RealMat &gridPoints, int derivativeLevel, RealMat &potential) {
        RealMat emptyMatrix(0, 0);
        computePRecHelper(parameterAngMom, parameters, coordinates, gridPoints, derivativeLevel, potential,
                          emptyMatrix);
    }

    /*!
     * \brief Runs a PME reciprocal space calculation, computing energies.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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

        filterAtomsAndBuildSplineCache(parameterAngMom, coordinates);
        auto realGrid = spreadParameters(parameterAngMom, parameters);
        Real energy;
        if (algorithmType_ == AlgorithmType::PME) {
            auto gridAddress = forwardTransform(realGrid);
            energy = convolveE(gridAddress);
        } else if (algorithmType_ == AlgorithmType::CompressedPME) {
            auto gridAddress = compressedForwardTransform(realGrid);
            energy = convolveE(gridAddress);
        } else {
            std::logic_error("Unknown algorithm in helpme::computeERec");
        }
        return energy;
    }

    /*!
     * \brief Runs a PME reciprocal space calculation, computing energies and forces.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
        // Spline derivative level bumped by 1, for energy gradients.
        filterAtomsAndBuildSplineCache(parameterAngMom + 1, coordinates);

        auto realGrid = spreadParameters(parameterAngMom, parameters);

        Real energy;
        if (algorithmType_ == AlgorithmType::PME) {
            auto gridAddress = forwardTransform(realGrid);
            energy = convolveE(gridAddress);
            auto potentialGrid = inverseTransform(gridAddress);
            probeGrid(potentialGrid, parameterAngMom, parameters, forces);
        } else if (algorithmType_ == AlgorithmType::CompressedPME) {
            auto gridAddress = compressedForwardTransform(realGrid);
            energy = convolveE(gridAddress);
            auto potentialGrid = compressedInverseTransform(gridAddress);
            probeGrid(potentialGrid, parameterAngMom, parameters, forces);
        } else {
            std::logic_error("Unknown algorithm in helpme::computeEFRec");
        }

        return energy;
    }

    /*!
     * \brief Runs a PME reciprocal space calculation, computing energies, forces and the virial.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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

        // Spline derivative level bumped by 1, for energy gradients.
        filterAtomsAndBuildSplineCache(parameterAngMom + 1, coordinates);

        auto realGrid = spreadParameters(parameterAngMom, parameters);

        Real energy;
        if (algorithmType_ == AlgorithmType::PME) {
            auto gridAddress = forwardTransform(realGrid);
            energy = convolveEV(gridAddress, virial);
            auto potentialGrid = inverseTransform(gridAddress);
            probeGrid(potentialGrid, parameterAngMom, parameters, forces, virial[0]);
        } else if (algorithmType_ == AlgorithmType::CompressedPME) {
            auto gridAddress = compressedForwardTransform(realGrid);
            Real *convolvedGrid;
            energy = convolveEV(gridAddress, convolvedGrid, virial);
            auto potentialGrid = compressedInverseTransform(convolvedGrid);
            probeGrid(potentialGrid, parameterAngMom, parameters, forces);
        } else {
            std::logic_error("Unknown algorithm in helpme::computeEFVRec");
        }

        return energy;
    }

    /*!
     * \brief Runs a PME reciprocal space calculation, computing energies, forces and the virial.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
     * \param inducedDipoles the induced dipoles in the order {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param polarizationType the method used to converged the induced dipoles.
     * \param coordinates the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
     * \param energy pointer to the variable holding the energy; this is incremented, not assigned.
     * \param forces a Nx3 matrix of the forces, ordered in memory as {Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,....FxN,FyN,FzN}.
     *        This matrix is incremented, not assigned.
     * \param virial a vector of length 6 containing the unique virial elements, in the order XX XY YY XZ YZ ZZ.
     *        This vector is incremented, not assigned.
     * \return the reciprocal space energy.
     */
    Real computeEFVRecIsotropicInducedDipoles(int parameterAngMom, const RealMat &parameters,
                                              const RealMat &inducedDipoles, PolarizationType polarizationType,
                                              const RealMat &coordinates, RealMat &forces, RealMat &virial) {
        sanityChecks(parameterAngMom, parameters, coordinates);
        if (parameterAngMom)
            throw std::runtime_error("Only point charges are allowed in computeEFVRecIsoPolarized() at the moment.");
        if (polarizationType != PolarizationType::Mutual)
            throw std::runtime_error("Only mutual (variation) optimized dipoles are supported at the moment.");

        size_t numAtoms = parameters.nRows();
        // Get the potential and field from the permanent moments
        RealMat potential(numAtoms, 10);
        RealMat combinedMultipoles(numAtoms, 4);
        for (int atom = 0; atom < parameters.nRows(); ++atom) {
            combinedMultipoles[atom][0] = parameters[atom][0];
            combinedMultipoles[atom][1] = inducedDipoles[atom][0];
            combinedMultipoles[atom][2] = inducedDipoles[atom][1];
            combinedMultipoles[atom][3] = inducedDipoles[atom][2];
        }

        computePVRec(1, combinedMultipoles, coordinates, coordinates, 2, potential, virial);

        double energy = 0;
        Real *virialPtr = virial.begin();
        Real &Vxx = virialPtr[0];
        Real &Vxy = virialPtr[1];
        Real &Vyy = virialPtr[2];
        Real &Vxz = virialPtr[3];
        Real &Vyz = virialPtr[4];
        Real &Vzz = virialPtr[5];
        for (int atom = 0; atom < numAtoms; ++atom) {
            const Real *dPhi = potential[atom];
            double charge = parameters[atom][0];
            Real *force = forces[atom];
            double phi = dPhi[0];
            double phiX = dPhi[1];
            double phiY = dPhi[2];
            double phiZ = dPhi[3];
            double phiXX = dPhi[4];
            double phiXY = dPhi[5];
            double phiYY = dPhi[6];
            double phiXZ = dPhi[7];
            double phiYZ = dPhi[8];
            double phiZZ = dPhi[9];
            const Real *mu = inducedDipoles[atom];
            energy += 0.5 * charge * phi;

            force[0] += phiXX * mu[0] + phiXY * mu[1] + phiXZ * mu[2];
            force[1] += phiXY * mu[0] + phiYY * mu[1] + phiYZ * mu[2];
            force[2] += phiXZ * mu[0] + phiYZ * mu[1] + phiZZ * mu[2];

            force[0] += charge * phiX;
            force[1] += charge * phiY;
            force[2] += charge * phiZ;

            Vxx += phiX * mu[0];
            Vxy += 0.5 * (phiX * mu[1] + phiY * mu[0]);
            Vyy += phiY * mu[1];
            Vxz += 0.5 * (phiX * mu[2] + phiZ * mu[0]);
            Vyz += 0.5 * (phiY * mu[2] + phiZ * mu[1]);
            Vzz += phiZ * mu[2];
        }
        return energy;
    }

    /*!
     * \brief Runs a full (direct and reciprocal space) PME calculation, computing the energy.  The direct space
     *        implementation here is not totally optimal, so this routine should primarily be used for testing and
     *        debugging.
     * \param includedList dense list of included atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN,jN.
     * \param excludedList dense list of excluded atom pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN, jN.
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
        sanityChecks(parameterAngMom, parameters, coordinates);

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
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.). \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
        sanityChecks(parameterAngMom, parameters, coordinates);

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
     * \param parameterAngMom the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for
     * quadrupoles, etc.).
     * \param parameters the list of parameters associated with each atom (charges, C6
     * coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
     * is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
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
        sanityChecks(parameterAngMom, parameters, coordinates);

        Real energy = computeEFVRec(parameterAngMom, parameters, coordinates, forces, virial);
        energy += computeESlf(parameterAngMom, parameters);
        energy += computeEFVDir(includedList, parameterAngMom, parameters, coordinates, forces, virial);
        energy += computeEFVAdj(excludedList, parameterAngMom, parameters, coordinates, forces, virial);
        return energy;
    }

    /*!
     * \brief setup initializes this object for a PME calculation using only threading.
     *        This may be called repeatedly without compromising performance.
     * \param rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive
     * dispersion).
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param splineOrder the order of B-spline; must be at least (2 + max. multipole order + deriv. level needed).
     * \param dimA the dimension of the FFT grid along the A axis.
     * \param dimB the dimension of the FFT grid along the B axis.
     * \param dimC the dimension of the FFT grid along the C axis.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param nThreads the maximum number of threads to use for each MPI instance; if set to 0 all available threads
     * are used.
     */
    void setup(int rPower, Real kappa, int splineOrder, int dimA, int dimB, int dimC, Real scaleFactor, int nThreads) {
        setupCalculationMetadata(rPower, kappa, splineOrder, dimA, dimB, dimC, dimA, dimB, dimC, scaleFactor, nThreads,
                                 0, NodeOrder::ZYX, 1, 1, 1);
    }

    /*!
     * \brief setupCompressed initializes this object for a compressed PME calculation using only threading.
     *        This may be called repeatedly without compromising performance.
     * \param rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive
     *        dispersion).
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param splineOrder the order of B-spline; must be at least (2 + max. multipole order + deriv. level needed).
     * \param dimA the dimension of the FFT grid along the A axis.
     * \param dimB the dimension of the FFT grid along the B axis.
     * \param dimC the dimension of the FFT grid along the C axis.
     * \param maxKA the maximum K value in the reciprocal sum along the A axis.
     * \param maxKB the maximum K value in the reciprocal sum along the B axis.
     * \param maxKC the maximum K value in the reciprocal sum along the C axis.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param nThreads the maximum number of threads to use for each MPI instance; if set to 0 all available threads
     * are used.
     */
    void setupCompressed(int rPower, Real kappa, int splineOrder, int dimA, int dimB, int dimC, int maxKA, int maxKB,
                         int maxKC, Real scaleFactor, int nThreads) {
        setupCalculationMetadata(rPower, kappa, splineOrder, dimA, dimB, dimC, maxKA, maxKB, maxKC, scaleFactor,
                                 nThreads, 0, NodeOrder::ZYX, 1, 1, 1);
    }
#if HAVE_MPI == 1
    /*!
     * \brief setupParallel initializes this object for a conventional PME calculation using MPI parallism
     *        and threading.  This may be called repeatedly without compromising performance.
     * \param rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive
     *        dispersion).
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param splineOrder the order of B-spline; must be at least (2 + max. multipole order + deriv. level needed).
     * \param dimA the dimension of the FFT grid along the A axis.
     * \param dimB the dimension of the FFT grid along the B axis.
     * \param dimC the dimension of the FFT grid along the C axis.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param nThreads the maximum number of threads to use for each MPI instance; if set to 0 all available threads
     *        are used.
     * \param communicator the MPI communicator for the reciprocal space calcultion, which should already be
     *        initialized.
     * \param numNodesA the number of nodes to be used for the A dimension.
     * \param numNodesB the number of nodes to be used for the B dimension.
     * \param numNodesC the number of nodes to be used for the C dimension.
     */
    void setupParallel(int rPower, Real kappa, int splineOrder, int dimA, int dimB, int dimC, Real scaleFactor,
                       int nThreads, const MPI_Comm &communicator, NodeOrder nodeOrder, int numNodesA, int numNodesB,
                       int numNodesC) {
        setupCalculationMetadata(rPower, kappa, splineOrder, dimA, dimB, dimC, dimA, dimB, dimC, scaleFactor, nThreads,
                                 (void *)&communicator, nodeOrder, numNodesA, numNodesB, numNodesC);
    }

    /*!
     * \brief setupCompressedParallel initializes this object for a compressed PME calculation using MPI parallism
     *        and threading.  This may be called repeatedly without compromising performance.
     * \param rPower the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive
     *        dispersion).
     * \param kappa the attenuation parameter in units inverse of those used to specify coordinates.
     * \param splineOrder the order of B-spline; must be at least (2 + max. multipole order + deriv. level needed).
     * \param dimA the dimension of the FFT grid along the A axis.
     * \param dimB the dimension of the FFT grid along the B axis.
     * \param dimC the dimension of the FFT grid along the C axis.
     * \param maxKA the maximum K value in the reciprocal sum along the A axis.
     * \param maxKB the maximum K value in the reciprocal sum along the B axis.
     * \param maxKC the maximum K value in the reciprocal sum along the C axis.
     * \param scaleFactor a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
     *        1 / [4 pi epslion0] for Coulomb calculations).
     * \param nThreads the maximum number of threads to use for each MPI instance; if set to 0 all available threads
     *        are used.
     * \param communicator the MPI communicator for the reciprocal space calcultion, which should already be
     *        initialized.
     * \param numNodesA the number of nodes to be used for the A dimension.
     * \param numNodesB the number of nodes to be used for the B dimension.
     * \param numNodesC the number of nodes to be used for the C dimension.
     */
    void setupCompressedParallel(int rPower, Real kappa, int splineOrder, int dimA, int dimB, int dimC, int maxKA,
                                 int maxKB, int maxKC, Real scaleFactor, int nThreads, const MPI_Comm &communicator,
                                 NodeOrder nodeOrder, int numNodesA, int numNodesB, int numNodesC) {
        setupCalculationMetadata(rPower, kappa, splineOrder, dimA, dimB, dimC, maxKA, maxKB, maxKC, scaleFactor,
                                 nThreads, (void *)&communicator, nodeOrder, numNodesA, numNodesB, numNodesC);
    }
#endif
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

typedef enum { Undefined = 0, XAligned = 1, ShapeMatrix = 2 } LatticeType;
typedef enum { /* Undefined comes from the above scope */ ZYX = 1 } NodeOrder;
typedef enum { Mutual = 0 } PolarizationType;

typedef struct PMEInstance PMEInstance;
extern struct PMEInstance *helpme_createD();
extern struct PMEInstance *helpme_createF();
extern void helpme_destroyD(struct PMEInstance *pme);
extern void helpme_destroyF(struct PMEInstance *pme);
extern void helpme_setupD(struct PMEInstance *pme, int rPower, double kappa, int splineOrder, int aDim, int bDim,
                          int cDim, double scaleFactor, int nThreads);
extern void helpme_setupF(struct PMEInstance *pme, int rPower, float kappa, int splineOrder, int aDim, int bDim,
                          int cDim, float scaleFactor, int nThreads);
extern void helpme_setup_compressedD(struct PMEInstance *pme, int rPower, double kappa, int splineOrder, int aDim,
                                     int bDim, int cDim, int maxKA, int maxKB, int maxKC, double scaleFactor,
                                     int nThreads);
extern void helpme_setup_compressedF(struct PMEInstance *pme, int rPower, float kappa, int splineOrder, int aDim,
                                     int bDim, int cDim, int maxKA, int maxKB, int maxKC, float scaleFactor,
                                     int nThreads);
#if HAVE_MPI == 1
extern void helpme_setup_parallelD(PMEInstance *pme, int rPower, double kappa, int splineOrder, int dimA, int dimB,
                                   int dimC, double scaleFactor, int nThreads, MPI_Comm communicator,
                                   NodeOrder nodeOrder, int numNodesA, int numNodesB, int numNodesC);
extern void helpme_setup_parallelF(PMEInstance *pme, int rPower, float kappa, int splineOrder, int dimA, int dimB,
                                   int dimC, float scaleFactor, int nThreads, MPI_Comm communicator,
                                   NodeOrder nodeOrder, int numNodesA, int numNodesB, int numNodesC);
extern void helpme_setup_compressed_parallelD(PMEInstance *pme, int rPower, double kappa, int splineOrder, int dimA,
                                              int dimB, int dimC, int maxKA, int maxKB, int maxKC, double scaleFactor,
                                              int nThreads, MPI_Comm communicator, NodeOrder nodeOrder, int numNodesA,
                                              int numNodesB, int numNodesC);
extern void helpme_setup_compressed_parallelF(PMEInstance *pme, int rPower, float kappa, int splineOrder, int dimA,
                                              int dimB, int dimC, int maxKA, int maxKB, int maxKC, float scaleFactor,
                                              int nThreads, MPI_Comm communicator, NodeOrder nodeOrder, int numNodesA,
                                              int numNodesB, int numNodesC);
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
extern void helpme_compute_P_recD(struct PMEInstance *pme, size_t nAtoms, int parameterAngMom, double *parameters,
                                  double *coordinates, size_t nGridPoints, double *gridPoints, int derivativeLevel,
                                  double *potential);
extern void helpme_compute_P_recF(struct PMEInstance *pme, size_t nAtoms, int parameterAngMom, float *parameters,
                                  float *coordinates, size_t nGridPoints, float *gridPoints, int derivativeLevel,
                                  float *potential);
#endif  // C++/C
#endif  // Header guard
