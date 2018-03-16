#include <mpi.h>
struct MPIHelper {
    static bool initialized_, canDelete_;
    int myRank_, numNodes_;
    void finalize() { canDelete_ = true; }
    MPIHelper() {
        if (!initialized_) {
            MPI_Init(NULL, NULL);
            MPI_Comm_size(MPI_COMM_WORLD, &numNodes_);
            MPI_Comm_rank(MPI_COMM_WORLD, &myRank_);
            initialized_ = true;
        }
    }
    ~MPIHelper() {
        if (canDelete_) MPI_Finalize();
    }
};
bool MPIHelper::initialized_ = false;
bool MPIHelper::canDelete_ = false;
