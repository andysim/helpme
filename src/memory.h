// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#ifndef _HELPME_MEMORY_H_
#define _HELPME_MEMORY_H_

#include <stdexcept>
#include <vector>

#include <fftw3.h>

namespace helpme {

/*!
 * \brief FFTWAllocator a class to handle aligned allocation of memory using the FFTW libraries.
 *        Code is adapted from http://www.josuttis.com/cppcode/myalloc.hpp.html.
 */
template <class T>
class FFTWAllocator {
   public:
    // type definitions
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // rebind allocator to type U
    template <class U>
    struct rebind {
        typedef FFTWAllocator<U> other;
    };

    // return address of values
    pointer address(reference value) const { return &value; }
    const_pointer address(const_reference value) const { return &value; }

    /* constructors and destructor
     * - nothing to do because the allocator has no state
     */
    FFTWAllocator() throw() {}
    FFTWAllocator(const FFTWAllocator&) throw() {}
    template <class U>
    FFTWAllocator(const FFTWAllocator<U>&) throw() {}
    ~FFTWAllocator() throw() {}
    FFTWAllocator& operator=(FFTWAllocator other) throw() {}
    template <class U>
    FFTWAllocator& operator=(FFTWAllocator<U> other) throw() {}

    // return maximum number of elements that can be allocated
    size_type max_size() const throw() { return std::numeric_limits<std::size_t>::max() / sizeof(T); }

    // allocate but don't initialize num elements of type T
    pointer allocate(size_type num, const void* = 0) { return static_cast<pointer>(fftw_malloc(num * sizeof(T))); }

    // initialize elements of allocated storage p with value value
    void construct(pointer p, const T& value) {
        // initialize memory with placement new
        new ((void*)p) T(value);
    }

    // destroy elements of initialized storage p
    void destroy(pointer p) {}

    // deallocate storage p of deleted elements
    void deallocate(pointer p, size_type num) { fftw_free(static_cast<void*>(p)); }
};

// return that all specializations of this allocator are interchangeable
template <class T1, class T2>
bool operator==(const FFTWAllocator<T1>&, const FFTWAllocator<T2>&) throw() {
    return true;
}
template <class T1, class T2>
bool operator!=(const FFTWAllocator<T1>&, const FFTWAllocator<T2>&) throw() {
    return false;
}

template <typename Real>
using vector = std::vector<Real, FFTWAllocator<Real>>;

}  // Namespace helpme

#endif  // Header guard
