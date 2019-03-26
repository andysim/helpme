#ifndef _HELPME_MIPP_WRAPPER_
#define _HELPME_MIPP_WRAPPER_

#ifdef NOMIPP

#define MAKEVEC(x) (x)
#define REALVEC Real
#define REALVECLEN 1
#define SIZETVEC size_t
#define SIZETVECLEN 1
#define STOREVEC(vec, loc) (loc) = (vec)
namespace mipp {
template <typename Real>
using vector = std::vector<Real>;
}

#else

#define MIPP_ALIGNED_LOADS
#include "mipp.h"
#define MAKEVEC(x) &(x)
#define REALVEC mipp::Reg<Real>
#define REALVECLEN mipp::N<Real>()
#define SIZETVEC mipp::Reg<size_t>
#define SIZETVECLEN mipp::N<size_t>()
#define STOREVEC(vec, loc) (vec).store(&(loc))

#endif

#endif  // Header guard
