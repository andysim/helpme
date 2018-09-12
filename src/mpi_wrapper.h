// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_MPI_WRAPPER_H_
#define _HELPME_MPI_WRAPPER_H_

#include <mpi.h>

#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace helpme {

/*!
 * \brief The MPITypes struct abstracts away the MPI_Datatype types for different floating point modes
 *        using templates to hide the details from the caller.
 */
template <typename Real>
struct MPITypes {
    MPI_Datatype realType_;
    MPI_Datatype complexType_;
    MPITypes() {
        throw std::runtime_error("MPI wrapper has not been implemented for the requested floating point type.");
    }
};

template <>
MPITypes<float>::MPITypes() : realType_(MPI_FLOAT), complexType_(MPI_C_COMPLEX) {}

template <>
MPITypes<double>::MPITypes() : realType_(MPI_DOUBLE), complexType_(MPI_C_DOUBLE_COMPLEX) {}

template <>
MPITypes<long double>::MPITypes() : realType_(MPI_LONG_DOUBLE), complexType_(MPI_C_LONG_DOUBLE_COMPLEX) {}

/*!
 * \brief The MPIWrapper struct is a lightweight C++ wrapper around the C MPI functions.  Its main
 *        purpose is to provide RAII semantics, ensuring that memory is correctly freed.  It also
 *        conveniently abstracts away the different MPI type descriptors for each floating point type.
 */
template <typename Real>
struct MPIWrapper {
    MPITypes<Real> types_;
    /// The MPI communicator instance to use for all reciprocal space work.
    MPI_Comm mpiCommunicator_;
    /// The total number of MPI nodes involved in reciprocal space work.
    int numNodes_;
    /// The MPI rank of this node.
    int myRank_;
    /// The number of nodes in the X direction.
    int numNodesX_;
    /// The number of nodes in the Y direction.
    int numNodesY_;
    /// The number of nodes in the Z direction.
    int numNodesZ_;

    void assertNodePartitioningValid(int numNodes, int numNodesX, int numNodesY, int numNodesZ) const {
        if (numNodes != numNodesX * numNodesY * numNodesZ)
            throw std::runtime_error(
                "Communicator world size does not match the numNodesX, numNodesY, numNodesZ passed in.");
    }

    MPIWrapper() : mpiCommunicator_(0), numNodes_(0), myRank_(0) {}
    MPIWrapper(const MPI_Comm& communicator, int numNodesX, int numNodesY, int numNodesZ)
        : numNodesX_(numNodesX), numNodesY_(numNodesY), numNodesZ_(numNodesZ) {
        if (MPI_Comm_dup(communicator, &mpiCommunicator_) != MPI_SUCCESS)
            throw std::runtime_error("Problem calling MPI_Comm_dup in MPIWrapper constructor.");
        if (MPI_Comm_size(mpiCommunicator_, &numNodes_) != MPI_SUCCESS)
            throw std::runtime_error("Problem calling MPI_Comm_size in MPIWrapper constructor.");
        if (MPI_Comm_rank(mpiCommunicator_, &myRank_) != MPI_SUCCESS)
            throw std::runtime_error("Problem calling MPI_Comm_rank in MPIWrapper constructor.");

        assertNodePartitioningValid(numNodes_, numNodesX, numNodesY, numNodesZ);
    }
    ~MPIWrapper() {
        if (mpiCommunicator_) MPI_Comm_free(&mpiCommunicator_);
    }

    /*!
     * \brief barrier wait for all members of this communicator to reach this point.
     */
    void barrier() {
        if (MPI_Barrier(mpiCommunicator_) != MPI_SUCCESS) throw std::runtime_error("Problem in MPI Barrier call!");
    }

    /*!
     * \brief split split this communicator into subgroups.
     * \param color the number identifying the subgroup the new communicator belongs to.
     * \param key the rank of the new communicator within the subgroup.
     * \return the new communicator.
     */
    std::unique_ptr<MPIWrapper> split(int color, int key) {
        std::unique_ptr<MPIWrapper> newWrapper(new MPIWrapper);
        if (MPI_Comm_split(mpiCommunicator_, color, key, &newWrapper->mpiCommunicator_) != MPI_SUCCESS)
            throw std::runtime_error("Problem calling MPI_Comm_split in MPIWrapper split.");
        if (MPI_Comm_size(newWrapper->mpiCommunicator_, &newWrapper->numNodes_) != MPI_SUCCESS)
            throw std::runtime_error("Problem calling MPI_Comm_size in MPIWrapper split.");
        if (MPI_Comm_rank(newWrapper->mpiCommunicator_, &newWrapper->myRank_) != MPI_SUCCESS)
            throw std::runtime_error("Problem calling MPI_Comm_rank in MPIWrapper split.");
        return newWrapper;
    }

    /*!
     * \brief allToAll perform alltoall communication within this communicator.
     * \param inBuffer the buffer containing input data.
     * \param outBuffer the buffer to send results to.
     * \param dimension the number of elements to be communicated.
     */
    void allToAll(std::complex<Real>* inBuffer, std::complex<Real>* outBuffer, int dimension) {
        if (MPI_Alltoall(inBuffer, 2 * dimension, types_.realType_, outBuffer, 2 * dimension, types_.realType_,
                         mpiCommunicator_) != MPI_SUCCESS)
            throw std::runtime_error("Problem encountered calling MPI alltoall.");
    }
    /*!
     * \brief allToAll perform alltoall communication within this communicator.
     * \param inBuffer the buffer containing input data.
     * \param outBuffer the buffer to send results to.
     * \param dimension the number of elements to be communicated.
     */
    void allToAll(Real* inBuffer, Real* outBuffer, int dimension) {
        if (MPI_Alltoall(inBuffer, dimension, types_.realType_, outBuffer, dimension, types_.realType_,
                         mpiCommunicator_) != MPI_SUCCESS)
            throw std::runtime_error("Problem encountered calling MPI alltoall.");
    }
    /*!
     * \brief reduce performs a reduction, with summation as the operation.
     * \param inBuffer the buffer containing input data.
     * \param outBuffer the buffer to send results to.
     * \param dimension the number of elements to be reduced.
     * \param node the node to reduce the result to (defaulted to zero).
     */
    void reduce(Real* inBuffer, Real* outBuffer, int dimension, int node = 0) {
        if (MPI_Reduce(inBuffer, outBuffer, dimension, types_.realType_, MPI_SUM, node, mpiCommunicator_) !=
            MPI_SUCCESS)
            throw std::runtime_error("Problem encountered calling MPI reduce.");
    }
    /*!
     * \brief reduceScatterBlock performs a reduction, with summation as the operation, then scatters to all nodes.
     * \param inBuffer the buffer containing input data.
     * \param outBuffer the buffer to send results to.
     * \param dimension the number of elements to be reduced on each node (currently must be the same on all nodes).
     */
    void reduceScatterBlock(Real* inBuffer, Real* outBuffer, int dimension) {
        if (MPI_Reduce_scatter_block(inBuffer, outBuffer, dimension, types_.realType_, MPI_SUM, mpiCommunicator_) !=
            MPI_SUCCESS)
            throw std::runtime_error("Problem encountered calling MPI reducescatter.");
    }
    /*!
     * \brief allGather broadcasts a chunk of data from each node to every other node.
     * \param inBuffer the buffer containing input data.
     * \param dimension the number of elements to be broadcast.
     * \param outBuffer the buffer to send results to.
     */
    void allGather(Real* inBuffer, Real* outBuffer, int dimension) {
        if (MPI_Allgather(inBuffer, dimension, types_.realType_, outBuffer, dimension, types_.realType_,
                          mpiCommunicator_) != MPI_SUCCESS)
            throw std::runtime_error("Problem encountered calling MPI allgather.");
    }

    /*!
     * \brief operator << a convenience wrapper around ostream, to inject node info.
     */
    friend std::ostream& operator<<(std::ostream& os, const MPIWrapper& obj) {
        os << "Node " << obj.myRank_ << " of " << obj.numNodes_ << ":" << std::endl;
        return os;
    }
};

// Adapter to allow piping of streams into unique_ptr-held object
template <typename Real>
std::ostream& operator<<(std::ostream& os, const std::unique_ptr<MPIWrapper<Real>>& obj) {
    os << *obj;
    return os;
}

// A convenience macro to guarantee that each node prints in order.
#define PRINT(out)                                                                                           \
    if (mpiCommunicator_) {                                                                                  \
        for (int node = 0; node < mpiCommunicator_->numNodes_; ++node) {                                     \
            std::cout.setf(std::ios::fixed, std::ios::floatfield);                                           \
            if (node == mpiCommunicator_->myRank_)                                                           \
                std::cout << mpiCommunicator_ << std::setw(18) << std::setprecision(10) << out << std::endl; \
            mpiCommunicator_->barrier();                                                                     \
        };                                                                                                   \
    } else {                                                                                                 \
        std::cout << std::setw(18) << std::setprecision(10) << out << std::endl;                             \
    }

}  // Namespace helpme
#endif  // Header guard
