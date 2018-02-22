#ifndef SPARSE_ARRAY2D_H
#define SPARSE_ARRAY2D_H

#include "array2d.h"

#include <vector>
#include <limits>


namespace TNT
{

template <typename T>
class SparseArray2D
{
    public:
        typedef std::pair<int, int> Coord;
        typedef std::map<Coord, T> Dict;
    
        SparseArray2D();
        SparseArray2D(int nr, int nc);
        SparseArray2D(const SparseArray2D &other);
        ~SparseArray2D() {};

        inline SparseArray2D& operator=(const SparseArray2D &rhs);

        inline T& operator()(int i);
        inline const T& operator()(int i) const;
        inline T& operator()(int i, int j);
        inline const T& operator()(int i, int j) const;

        inline int dim1() const;
        inline int dim2() const;
        inline int size() const;
        
        inline int sub2ind(const Coord &subs) const;
        inline Coord ind2sub(const int ind) const;

        inline void reshape(int nr, int nc);
        inline Array2D<T> full() const;
        
        
        template <typename S>
        friend SparseArray2D<S> getsub(const SparseArray2D<S> &A, int r0, int r1, int c0, int c1);
        
        template <typename S>
        friend void setsub(SparseArray2D<S> &A, int r0, int r1, int c0, int c1, const SparseArray2D<S> &B);
        
        template <typename S, typename UnaryPred> 
        friend std::vector<int> find(const SparseArray2D<S> &v, const UnaryPred pred);
        
        template <typename S, typename UnaryPred> 
        friend SparseArray2D<S> dotor(const SparseArray2D<S> &A, const SparseArray2D<S> &B);
        
        template <typename S>
        friend Array2D<S> sum(const SparseArray2D<S> &A, const std::string &dim);
       
        template <typename S>
        friend SparseArray2D<S> dotor(const SparseArray2D<S> &A, const SparseArray2D<S> &B);

    private:
        int nrows;
        int ncols;

        std::map<Coord, T> data;
        T m_zero;
};


template <typename T> Array2D<T> sum(const SparseArray2D<T> &A, const std::string &dim="");


//////////////////////////////////////////////////////////////////////////////
// template implementation
//////////////////////////////////////////////////////////////////////////////

template <typename T>
SparseArray2D<T>::SparseArray2D() : nrows(0), ncols(0), m_zero(0)
{}

template <typename T>
SparseArray2D<T>::SparseArray2D(int nr, int nc) : nrows(nr), ncols(nc), m_zero(0)
{
    assert(nr >= 0);
    assert(nc >= 0);
}

template <typename T>
T& SparseArray2D<T>::operator()(int i)
{
    Coord subs = ind2sub(i);
    return this->operator()(subs.first, subs.second);
}

template <typename T>
const T& SparseArray2D<T>::operator()(int i) const
{
    Coord subs = ind2sub(i);
    return this->operator()(subs.first, subs.second);
}

template <typename T>
T& SparseArray2D<T>::operator()(int i, int j)
{
#ifdef BOUNDS_CHECK
    assert((i < nrows) && (j < ncols));
#endif
    typename Dict::iterator it = data.find(Coord(i, j));
    if(it!=data.end())
    {
        return it->second;
    }
    else
    {
        data[Coord(i, j)] = 0;
        return data[Coord(i, j)];
    }    
}

template <typename T>
const T& SparseArray2D<T>::operator()(int i, int j) const
{
#ifdef BOUNDS_CHECK
    assert((i < nrows) && (j < ncols));
#endif
    typename Dict::const_iterator it = data.find(Coord(i, j));
    if(it!=data.end())
    {
        return it->second;
    }
    else
    {
        return m_zero;
    } 
}

template <typename T>
int SparseArray2D<T>::dim1() const
{
    return nrows;
}

template <typename T>
int SparseArray2D<T>::dim2() const
{
    return ncols;
}

template <typename T>
int SparseArray2D<T>::size() const
{
    return data.size();
}

template <typename T>
int SparseArray2D<T>::sub2ind(const Coord &subs) const
{
#ifdef BOUNDS_CHECK
    assert((subs.first < nrows) && (subs.second < ncols));
#endif
    return subs.first+nrows*subs.second;
}

template <typename T>
typename SparseArray2D<T>::Coord SparseArray2D<T>::ind2sub(const int ind) const
{
    Coord subs(ind%nrows, ind/nrows);
    return subs;
}

template <typename T>
inline Array2D<T> SparseArray2D<T>::full() const
{
    Array2D<T> A(dim1(), dim2(), 0);
    for(typename SparseArray2D<T>::Dict::const_iterator it = data.begin(); it != data.end(); ++it)
    {
        A(it->first.first, it->first.second) = it->second;
    }
    
    return A;
}

// B = A(r0:r1, c0:c1)
template <typename T>
SparseArray2D<T> getsub(const SparseArray2D<T> &A, int r0, int r1, int c0, int c1)
{
    assert(r0 >= 0 && c0 >= 0);
    assert(r1 < A.dim1() && c1 < A.dim2());
    
    int m = r1-r0+1;
    int n = c1-c0+1;
    assert(m > 0 && n > 0);

    SparseArray2D<T> B(m, n);
    for(typename SparseArray2D<T>::Dict::const_iterator it = A.data.begin(); it != A.data.end(); ++it)
    {
        const int i = it->first.first;
        const int j = it->first.second;
        if(i>=r0 && i<=r1 && j>=c0 && j<=c1)
        {
            B(i-r0, j-c0) = it->second;
        }
    }
    
    return B;
}

// B = A(r0:r1, :)
template <typename T>
SparseArray2D<T> getrows(const SparseArray2D<T> &A, int r0, int r1)
{
    return getsub(A, r0, r1, 0, A.dim2()-1);
}

// B = A(:, c0:c1)
template <typename T>
SparseArray2D<T> getcols(const SparseArray2D<T> &A, int c0, int c1)
{
    return getsub(A, 0, A.dim1()-1, c0, c1);
}

// B = A(r, :)
template <typename T>
SparseArray2D<T> getrow(const SparseArray2D<T> &A, int r)
{
    return getrows(A, r, r);
}

// B = A(:, c)
template <typename T>
SparseArray2D<T> getcol(const SparseArray2D<T> &A, int c)
{
    return getcols(A, c, c);
}

// A(r0:r1, c0:c1) = B
template <typename T>
void setsub(SparseArray2D<T> &A, int r0, int r1, int c0, int c1, const SparseArray2D<T> &B)
{    
    assert(r0 >= 0 && c0 >= 0);
    assert(r1 < A.dim1() && c1 < A.dim2());
    
    int m = r1-r0+1;
    int n = c1-c0+1;
    assert(m > 0 && n > 0);
    assert(m == B.dim1() && n == B.dim2());

    typename SparseArray2D<T>::Dict::iterator it = A.data.begin();
    while(it != A.data.end())
    {
        const int i = it->first.first;
        const int j = it->first.second;
        if(i>=r0 && i<=r1 && j>=c0 && j<=c1)
        {
            A.data.erase(it++);
        }
        else
        {
            ++it;
        }
    }
    
    for(typename SparseArray2D<T>::Dict::const_iterator it = B.data.begin(); it != B.data.end(); ++it)
    {
        const int i = it->first.first;
        const int j = it->first.second;
        A(i+r0, j+c0) = it->second;
    }
}

// A(r0:r1, :) = B
template <typename T>
void setrows(SparseArray2D<T> &A, int r0, int r1, const SparseArray2D<T> &B)
{
    return setsub(A, r0, r1, 0, A.dim2()-1, B);
}

// A(:, c0:c1) = B
template <typename T>
void setcols(SparseArray2D<T> &A, int c0, int c1, const SparseArray2D<T> &B)
{
    return setsub(A, 0, A.dim1()-1, c0, c1, B);
}

// A(r, :) = B
template <typename T>
void setrow(SparseArray2D<T> &A, int r, const SparseArray2D<T> &B)
{
    return setrows(A, r, r, B);
}

// A(:, c) = B
template <typename T>
void setcol(SparseArray2D<T> &A, int c, const SparseArray2D<T> &B)
{
    return setcols(A, c, c, B);
}

// B = sum(A, dim)
template <typename T>
Array2D<T> sum(const SparseArray2D<T> &A, const std::string &dim)
{
    if(A.dim1() == 0 && A.dim2() == 0)
        return Array2D<T>();

    Array2D<T> B;
    if(((A.dim1() == 1) || (A.dim2() == 1)) && (dim == ""))
    {
        B = Array2D<T>(1, 1, T(0));
        for(typename SparseArray2D<T>::Dict::const_iterator it = A.data.begin(); it != A.data.end(); ++it)
            B(0) += it->second;
    }
    else if((dim == "rows") || (dim == ""))
    {
        B = Array2D<T>(1, A.dim2(), T(0));
        for(typename SparseArray2D<T>::Dict::const_iterator it = A.data.begin(); it != A.data.end(); ++it)
        {
            const int j = it->first.second;
            B(j) += it->second;
        }
    }
    else if(dim == "cols")
    {
        B = Array2D<T>(A.dim1(), 1, T(0));
        for(typename SparseArray2D<T>::Dict::const_iterator it = A.data.begin(); it != A.data.end(); ++it)
        {
            const int i = it->first.first;
            B(i) += it->second;
        }
    }
    else
        assert(false);

    return B;
}

// find function
template <typename T, typename UnaryPred>
std::vector<int> find(const SparseArray2D<T> &v, const UnaryPred pred)
{
    std::vector<int> inds;
    for(typename SparseArray2D<T>::Dict::const_iterator it = v.data.begin(); it != v.data.end(); ++it)
        if(pred(it->second))
            inds.push_back(v.sub2ind(it->first));

    return inds;
}

// A = A | B
template <typename T>
SparseArray2D<T> dotor(const SparseArray2D<T> &A, const SparseArray2D<T> &B)
{
	assert(A.dim1() == B.dim1());
    assert(A.dim2() == B.dim2());

    SparseArray2D<T> C(A.dim1(), A.dim2());
    for(typename SparseArray2D<T>::Dict::const_iterator it = A.data.begin(); it != A.data.end(); ++it)
    {
        C.data[it->first] = C.data[it->first]|it->second;
    }
    for(typename SparseArray2D<T>::Dict::const_iterator it = B.data.begin(); it != B.data.end(); ++it)
    {
        C.data[it->first] = C.data[it->first]|it->second;
    }

    return C;
}


}

#endif

