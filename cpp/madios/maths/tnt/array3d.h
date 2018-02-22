#ifndef ARRAY3D_H
#define ARRAY3D_H

#include "array1d.h"
#include "array2d.h"


namespace TNT
{

template <typename T>
class Array3D
{
    public:
        Array3D();
        Array3D(int nr, int nc, int nm);
        Array3D(int nr, int nc, int nm, const T &a);
        Array3D(int nr, int nc, int nm, const T *a);
        Array3D(const Array3D &other);
        ~Array3D();

        inline Array3D& operator=(const Array3D &rh);
        inline Array3D& operator=(const T &a);

        inline Array3D& set(const Array3D<bool> &flags, const T &a);

        inline T& operator()(int i);
        inline const T& operator()(int i) const;
        inline T& operator()(int i, int j, int k);
        inline const T& operator()(int i, int j, int k) const;

        inline int dim1() const;
        inline int dim2() const;
        inline int dim3() const;
        inline int size() const;

        inline Array1D<T>& vector();
        inline const Array1D<T>& vector() const;

        inline T* vals();
        inline T* vals() const;

    private:
        int nrows;
        int ncols;
        int nmats;

        Array1D<T> data;
};


// IO functions
template <typename T> std::ostream& operator<<(std::ostream &s, const Array3D<T> &A);
template <typename T> std::istream& operator>>(std::istream &s, Array3D<T> &A);
template <typename T> void saveArray3D(const Array3D<T> &A, const std::string &filename);
template <typename T> Array3D<T> loadArray3D(const std::string &filename);

// convert
template <typename T, typename S> Array2D<S> convert(Array2D<T> &B, const Array2D<S> &A);

// additional functions and operators
template <typename T> Array3D<bool> operator>(const Array3D<T> &A, const T &s);
template <typename T> Array2D<T> squeeze(const Array3D<T> &A);
template <typename T> Array3D<T> inflate(const Array2D<T> &A, const std::string &dim);

// tensor operators
template <typename T> Array3D<T> operator-(const Array3D<T> &A, const Array3D<T> &B);
template <typename T> Array3D<T>&  operator+=(Array3D<T> &A, const Array3D<T> &B);
template <typename T> Array3D<T> operator-(const T &s, const Array3D<T> &A);
template <typename T> Array3D<T> operator*(const Array3D<T> &A, const T &s);
template <typename T> Array3D<T> operator/(const Array3D<T> &A, const T &s);
template <typename T> Array3D<T> sqrt(const Array3D<T> &A);
template <typename T> Array3D<T> fabs(const Array3D<T> &A);
template <typename T> Array3D<T> sum3(const Array3D<T> &A);



//////////////////////////////////////////////////////////////////////////////
// template implementation
//////////////////////////////////////////////////////////////////////////////

template <typename T>
Array3D<T>::Array3D() : nrows(0), ncols(0), nmats(0)
{}

template <typename T>
Array3D<T>::Array3D(int nr, int nc, int nm) : nrows(nr), ncols(nc), nmats(nm), data(nr*nc*nm)
{
    assert(nr >= 0);
    assert(nc >= 0);
    assert(nm >= 0);
}

template <typename T>
Array3D<T>::Array3D(int nr, int nc, int nm, const T &val) : nrows(nr), ncols(nc), nmats(nm), data(nr*nc*nm, val)
{
    assert(nr >= 0);
    assert(nc >= 0);
    assert(nm >= 0);
}


template <typename T>
Array3D<T>::Array3D(int nr, int nc, int nm, const T *a) : nrows(nr), ncols(nc), nmats(nm), data(nr*nc*nm, a)
{
    assert(nr >= 0);
    assert(nc >= 0);
    assert(nm >= 0);
}

template <typename T>
Array3D<T>::Array3D(const Array3D<T> &other) : nrows(other.nrows), ncols(other.ncols), nmats(other.nmats), data(other.data)
{}

template <typename T>
Array3D<T>::~Array3D()
{}


template <typename T>
Array3D<T>& Array3D<T>::operator=(const Array3D &rhs)
{
    data = rhs.data;
    nrows = rhs.nrows;
    ncols = rhs.ncols;
    nmats = rhs.nmats;

    return *this;
}

template <typename T>
Array3D<T>& Array3D<T>::operator=(const T &a)
{
    data = a;

    return *this;
}

template <typename T>
Array3D<T>& Array3D<T>::set(const Array3D<bool> &flags, const T &a)
{
    assert(dim1() == flags.dim1());
    assert(dim2() == flags.dim2());
    assert(dim3() == flags.dim3());

    data.set(flags.vector(), a);

    return *this;
}

template <typename T>
T& Array3D<T>::operator()(int i)
{
    return data[i];
}

template <typename T>
const T& Array3D<T>::operator()(int i) const
{
    return data[i];
}

template <typename T>
T& Array3D<T>::operator()(int i, int j, int k)
{
#ifdef TNT_BOUNDS_CHECK
    assert(i >= 0);
    assert(i < nrows);
    assert(j >= 0);
    assert(j < ncols);
    assert(k >= 0);
    assert(k < nmats);
#endif
    return data[i+j*nrows+k*nrows*ncols];
}

template <typename T>
const T& Array3D<T>::operator()(int i, int j, int k) const
{
#ifdef TNT_BOUNDS_CHECK
    assert(i >= 0);
    assert(i < nrows);
    assert(j >= 0);
    assert(j < ncols);
    assert(k >= 0);
    assert(k < nmats);
#endif
    return data[i+j*nrows+k*nrows*ncols];
}


template <typename T>
int Array3D<T>::dim1() const
{
    return nrows;
}

template <typename T>
int Array3D<T>::dim2() const
{
    return ncols;
}

template <typename T>
int Array3D<T>::dim3() const
{
    return nmats;
}

template <typename T>
int Array3D<T>::size() const
{
    return data.size();
}

template <typename T>
Array1D<T>& Array3D<T>::vector()
{
    return data;
}

template <typename T>
const Array1D<T>& Array3D<T>::vector() const
{
    return data;
}

template <typename T>
T* Array3D<T>::vals()
{
    return data.vals();
}

template <typename T>
T* Array3D<T>::vals() const
{
    return data.vals();
}


// IO functions
template <typename T>
std::ostream& operator<<(std::ostream &s, const Array3D<T> &A)
{
    int M=A.dim1();
    int N=A.dim2();
    int K=A.dim3();

    s << M << " " << N << " " << K << "\n";

    for (int k=0; k<K; k++)
    {
        for (int i=0; i<M; i++)
        {
            for (int j=0; j<N; j++)
                s << A(i, j, k) << " ";
            s << endl;
        }
        s << endl;
    }


    return s;
}

template <typename T>
std::istream& operator>>(std::istream &s, Array3D<T> &A)
{

    int M, N, K;

    s >> M >> N >> K;

    A = Array3D<T>(M,N,K);
    for (int k=0; k<K; k++)
        for (int i=0; i<M; i++)
            for (int j=0; j<N; j++)
                s >>  A(i, j, k);

    return s;
}

template <typename T>
void saveArray3D(const Array3D<T> &A, const std::string &filename)
{
    std::ofstream outfile;
    outfile.open(filename.c_str(), std::ostream::out);
    assert(outfile.is_open());

    outfile << A;
}

template <typename T>
Array3D<T> loadArray3D(const std::string &filename)
{
    std::ifstream infile;
    infile.open(filename.c_str(), std::ifstream::in);
    assert(infile.is_open());

    Array3D<T> A;
    infile >> A;

    return A;
}


// convert
template <typename T, typename S> 
void convert(Array3D<T> &B, const Array3D<S> &A)
{
    assert(B.dim1() == A.dim1());
    assert(B.dim2() == A.dim2());
    assert(B.dim3() == A.dim3());

    for(int i = 0; i < A.size(); i++)
        B(i) = static_cast<T>(A(i));
}



//////////////////////////////////////////////////////////////////////////////
// additional functions and operators
//////////////////////////////////////////////////////////////////////////////

// B = A > s
template <typename T>
Array3D<bool> operator>(const Array3D<T> &A, const T &s)
{
    Array3D<bool> B(A.dim1(), A.dim2(), A.dim3());
    greaterThan(B.vector(), A.vector(), s);

    return B;
}

// squeeze singleton dimension
template <typename T>
Array2D<T> squeeze(const Array3D<T> &A)
{
    if(A.dim1() == A.size() || A.dim2() == A.size() || A.dim3() == A.size())
        return Array2D<T>(A.size(), 1, A.vals());
    else if(A.dim1() == 1)
        return Array2D<T>(A.dim2(), A.dim3(), A.vals());
    else if(A.dim2() == 1)
        return Array2D<T>(A.dim1(), A.dim3(), A.vals());
    else if(A.dim3() == 1)
        return Array2D<T>(A.dim1(), A.dim2(), A.vals());
    else
        assert(false);
       
    return Array2D<T>();
}

// inflate singleton dimension
template <typename T>
Array3D<T> inflate(const Array2D<T> &A, const std::string &dim)
{
    if(dim == "rows")
        return Array3D<T>(1, A.dim1(), A.dim2(), A.vals());
    else if(dim == "cols")
        return Array3D<T>(A.dim1(), 1, A.dim2(), A.vals());
    else if(dim == "mats" || dim == "") 
        return Array3D<T>(A.dim1(), A.dim2(), 1, A.vals());
    else
        assert(false);

    return Array3D<T>();
}



//////////////////////////////////////////////////////////////////////////////
// tensor operators
//////////////////////////////////////////////////////////////////////////////

// C = A - B
template <typename T>
Array3D<T> operator-(const Array3D<T> &A, const Array3D<T> &B)
{
    assert(A.dim1() == B.dim1());
    assert(A.dim2() == B.dim2());
    assert(A.dim3() == B.dim3());

    Array3D<T> C(A.dim1(), A.dim2(), A.dim3());
    sub(C.vector(), A.vector(), B.vector());

    return C;
}

// A = A + B
template <typename T>
Array3D<T>&  operator+=(Array3D<T> &A, const Array3D<T> &B)
{
    assert(A.dim1() == B.dim1());
    assert(A.dim2() == B.dim2());
    assert(A.dim3() == B.dim3());	

    add(A.vector(), A.vector(), B.vector());

    return A;
}

// B = s - A
template <typename T>
Array3D<T> operator-(const T &s, const Array3D<T> &A)
{
    Array3D<T> B(A.dim1(), A.dim2(), A.dim3());
    sub(B.vector(), s, A.vector());

    return B;
}

// B = A * s
template <typename T>
Array3D<T> operator*(const Array3D<T> &A, const T &s)
{
    Array3D<T> B(A.dim1(), A.dim2(), A.dim3());
    dotmult(B.vector(), A.vector(), s);

    return B;
}

// B = A ./ s
template <typename T>
Array3D<T> operator/(const Array3D<T> &A, const T &s)
{
	return A*(1.0/s);
}

// B = sqrt(A)
template <typename T>
Array3D<T> sqrt(const Array3D<T> &A)
{
    Array3D<T> B(A.dim1(), A.dim2(), A.dim3());
    sqrt(B.vector(), A.vector());

    return B;
}

// B = fabs(A)
template <typename T>
Array3D<T> fabs(const Array3D<T> &A)
{
    Array3D<T> B(A.dim1(), A.dim2(), A.dim3());
    fabs(B.vector(), A.vector());

    return B;
}

// B = sum3(A)
template <typename T>
Array3D<T> sum3(const Array3D<T> &A)
{
    assert(A.dim1() > 0);
    assert(A.dim2() > 0);
    assert(A.dim3() > 0);

    const int nm = A.dim1()*A.dim2();

    Array3D<T> B(A.dim1(), A.dim2(), 1);

    T *psource = A.vals();
    T *ptarget = B.vals();
    for(int k = 0; k < A.dim3(); k++)
    {
        ptarget = B.vals();
        for(int i = 0; i < nm; i++)
        {
            *ptarget += *psource;
            psource++;
            ptarget++;
        }
    }

    return B;
}


}


#endif

