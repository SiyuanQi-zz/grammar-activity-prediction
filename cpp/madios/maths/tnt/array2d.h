#ifndef ARRAY2D_H
#define ARRAY2D_H

#include "array1d.h"

#include <vector>
#include <limits>


namespace TNT
{

template <typename T>
class Array2D
{
    public:
        typedef std::pair<int, int> Coord;
        
        Array2D();
        Array2D(int nr, int nc);
        Array2D(int nr, int nc, const T &a);
        Array2D(int nr, int nc, const T *a);
        Array2D(const Array2D &other);
        ~Array2D();

        inline Array2D& operator=(const Array2D &rhs);
        inline Array2D& operator=(const T &a);

        inline Array2D& set(const Array2D<bool> &flags, const T &a);

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
        inline const Array2D<T>& copy() const; // for compatibility with original TNT

        inline Array1D<T>& vector();
        inline const Array1D<T>& vector() const;

        inline T* vals();
        inline T* vals() const;

    private:
        int nrows;
        int ncols;

        Array1D<T> data;
};


// IO functions
template <typename T> std::ostream& operator<<(std::ostream &s, const Array2D<T> &A);
template <typename T> std::istream& operator>>(std::istream &s, Array2D<T> &A);
template <typename T> void saveArray2D(const Array2D<T> &A, const std::string &filename);
template <typename T> Array2D<T> loadArray2D(const std::string &filename);

// subarray functions
template <typename T> Array2D<T> getsub(const Array2D<T> &A, int r0, int r1, int c0, int c1);
template <typename T> Array2D<T> getrows(const Array2D<T> &A, int r0, int r1);
template <typename T> Array2D<T> getcols(const Array2D<T> &A, int c0, int c1);
template <typename T> Array2D<T> getrow(const Array2D<T> &A, int r);
template <typename T> Array2D<T> getcol(const Array2D<T> &A, int c);
template <typename T> void setsub(Array2D<T> &A, int r0, int r1, int c0, int c1, const Array2D<T> &B);
template <typename T> void setrows(Array2D<T> &A, int r0, int r1, const Array2D<T> &B);
template <typename T> void setcols(Array2D<T> &A, int c0, int c1, const Array2D<T> &B);
template <typename T> void setrow(Array2D<T> &A, int r, const Array2D<T> &B);
template <typename T> void setcol(Array2D<T> &A, int c, const Array2D<T> &B);

// manipulation functions
template <typename T> Array2D<T> transpose(const Array2D<T> &A);
template <typename T> Array2D<T> repmat(const Array2D<T> &A, unsigned int m, unsigned int n);
template <typename T> Array2D<T> reshape(const Array2D<T> &A, unsigned int m, unsigned int n);

// diag operators
template <typename T> Array2D<T> diag(const Array2D<T> &A);

// convert
template <typename T, typename S> void convert(Array2D<T> &B, const Array2D<S> &A);

// additional functions and operators
template <typename T> Array2D<bool> operator>(const Array2D<T> &A, const T &s);
template <typename T> Array2D<T> eye(int dim);
template <typename T> Array2D<T> zeros(int dim1, int dim2);
template <typename T> Array2D<T> zeros(int dim);
template <typename T> Array2D<T> ones(int dim1, int dim2);
template <typename T> Array2D<T> ones(int dim);
template <typename T> bool square(const Array2D<T> &A);

// matrix elementwise operators
template <typename T> Array2D<T> operator+(const Array2D<T> &A, const Array2D<T> &B);
template <typename T> Array2D<T> operator-(const Array2D<T> &A, const Array2D<T> &B);
template <typename T> Array2D<T> dotmult(const Array2D<T> &A, const Array2D<T> &B);
template <typename T> Array2D<T> dotdiv(const Array2D<T> &A, const Array2D<T> &B);
template <typename T> Array2D<T>& operator+=(Array2D<T> &A, const Array2D<T> &B);

// matrix matrix operators
template <typename T> Array2D<T> operator*(const Array2D<T> &A, const Array2D<T> &B);

// matrix vector operators
template <typename T> Array1D<T> operator*(const Array2D<T> &A, const Array1D<T> &b);
template <typename T> Array1D<T> operator*(const Array1D<T> &a, const Array2D<T> &B);

// matrix scalar operators
template <typename T> Array2D<T> operator-(const Array2D<T> &A);
template <typename T> Array2D<T> operator+(const Array2D<T> &A, const T &s);
template <typename T> Array2D<T> operator-(const Array2D<T> &A, const T &s);
template <typename T> Array2D<T> operator-(const T &s, const Array2D<T> &A);
template <typename T> Array2D<T> operator*(const Array2D<T> &A, const T &s);
template <typename T> Array2D<T> operator*(const T &s, const Array2D<T> &A);
template <typename T> Array2D<T> operator/(const Array2D<T> &A, const T &s);
template <typename T> Array2D<T> operator/(const T &s, const Array2D<T> &A);

// trace and more utility functions
template <typename T> T trace(const Array2D<T> &A);
template <typename T> T trace_matmult(const Array2D<T> &A, const Array2D<T> &B);

// sum, prod, cumsum, mean and max operators
template <typename T> T sum_vec(const Array2D<T> &A);
template <typename T> Array2D<T> sum(const Array2D<T> &A, const std::string &dim="");
template <typename T> Array2D<T> cumsum(const Array2D<T> &A, const std::string &dim="");
template <typename T> Array2D<T> mean(const Array2D<T> &A, const std::string &dim="");
template <typename T> Array1D<T> max(const Array2D<T> &A, const std::string &dim="");
template <typename T> Array1D<T> min(const Array2D<T> &A, const std::string &dim="");

// variance operators
template <typename T> Array2D<T> cov(const Array2D<T> &A);
template <typename T> Array2D<T> var(const Array2D<T> &A);
template <typename T> Array2D<T> cov(const Array1D<T> &a);

// distance operators
template <typename T> T mahalanobis(const Array2D<T> &A, const Array1D<T> &b);
template <typename T> Array2D<T> mahalanobis(const Array2D<T> &A, const Array2D<T> &B);
template <typename T> Array2D<T> dist2(const Array2D<T> &A, const Array2D<T> &B);

// overloaded maths operators
template <typename T> Array2D<T> log(const Array2D<T> &A);
template <typename T> Array2D<T> exp(const Array2D<T> &A);
template <typename T> Array2D<T> sqrt(const Array2D<T> &A);
template <typename T> Array2D<T> fabs(const Array2D<T> &A);

// assert functions
template <typename T> bool isSquareMatrix(const Array2D<T> &A);

// normalisation functions
template <typename T> Array2D<T> makeHomogeneous(const Array2D<T> &A);
template <typename T> Array2D<T> normaliseCols(const Array2D<T> &A);

//////////////////////////////////////////////////////////////////////////////
// template implementation
//////////////////////////////////////////////////////////////////////////////

template <typename T>
Array2D<T>::Array2D() : nrows(0), ncols(0)
{}

template <typename T>
Array2D<T>::Array2D(int nr, int nc) : nrows(nr), ncols(nc), data(nr*nc)
{
    assert(nr >= 0);
    assert(nc >= 0);
}

template <typename T>
Array2D<T>::Array2D(int nr, int nc, const T &a) : nrows(nr), ncols(nc), data(nr*nc, a)
{
    assert(nr >= 0);
    assert(nc >= 0);
}

template <typename T>
Array2D<T>::Array2D(int nr, int nc, const T *a) : nrows(nr), ncols(nc), data(nr*nc, a)
{
    assert(nr >= 0);
    assert(nc >= 0);
}

template <typename T>
Array2D<T>::Array2D(const Array2D &other): nrows(other.nrows), ncols(other.ncols), data(other.data)
{}

template <typename T>
Array2D<T>::~Array2D()
{}

template <typename T>
Array2D<T>& Array2D<T>::operator=(const Array2D &rhs)
{
    nrows = rhs.nrows;
    ncols = rhs.ncols;
    data = rhs.data;

    return *this;
}

template <typename T>
Array2D<T>& Array2D<T>::operator=(const T &a)
{
    data = a;

    return *this;
}

template <typename T>
Array2D<T>& Array2D<T>::set(const Array2D<bool> &flags, const T &a)
{
    assert(dim1() == flags.dim1());
    assert(dim2() == flags.dim2());

    data.set(flags.vector(), a);

    return *this;
}

template <typename T>
T& Array2D<T>::operator()(int i)
{
    return data[i];
}

template <typename T>
const T& Array2D<T>::operator()(int i) const
{
    return data[i];
}

template <typename T>
T& Array2D<T>::operator()(int i, int j)
{
#ifdef BOUNDS_CHECK
    assert((i < nrows) && (j < ncols));
#endif
    return data[i+nrows*j];
}

template <typename T>
const T& Array2D<T>::operator()(int i, int j) const
{
#ifdef BOUNDS_CHECK
    assert((i < nrows) && (j < ncols));
#endif
    return data[i+nrows*j];
}

template <typename T>
int Array2D<T>::dim1() const
{
    return nrows;
}

template <typename T>
int Array2D<T>::dim2() const
{
    return ncols;
}

template <typename T>
int Array2D<T>::size() const
{
    return data.size();
}

template <typename T>
int Array2D<T>::sub2ind(const Coord &subs) const
{
#ifdef BOUNDS_CHECK
    assert((subs.first < nrows) && (subs.second < ncols));
#endif
    return subs.first+nrows*subs.second;
}

template <typename T>
typename Array2D<T>::Coord Array2D<T>::ind2sub(const int ind) const
{
    Coord subs(ind%nrows, ind/nrows);
    return subs;
}

template <typename T>
void Array2D<T>::reshape(int nr, int nc)
{
    assert((nrows*ncols) == (nr*nc));

    nrows = nr;
    ncols = nc;
}

template <typename T>
const Array2D<T>& Array2D<T>::copy() const
{
    return *this;
}

template <typename T>
Array1D<T>& Array2D<T>::vector()
{
    return data;
}

template <typename T>
const Array1D<T>& Array2D<T>::vector() const
{
    return data;
}

template <typename T>
T* Array2D<T>::vals()
{
    return data.vals();
}

template <typename T>
T* Array2D<T>::vals() const
{
    return data.vals();
}



//////////////////////////////////////////////////////////////////////////////
// utility functions implementation
//////////////////////////////////////////////////////////////////////////////

// IO functions
template <typename T>
std::ostream& operator<<(std::ostream &s, const Array2D<T> &A)
{
    int M=A.dim1();
    int N=A.dim2();

    s << M << " " << N << "\n";
    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
            s << A(i, j) << " ";
        s << "\n";
    }

    return s;
}

template <typename T>
std::istream& operator>>(std::istream &s, Array2D<T> &A)
{
    int M, N;
    s >> M >> N;

    assert(M >= 0 && N >= 0);

    A = Array2D<T>(M,N);
    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++)
            s >>  A(i, j);

    return s;
}

template <typename T>
void saveArray2D(const Array2D<T> &A, const std::string &filename)
{
    std::ofstream outfile;
    outfile.open(filename.c_str(), std::ostream::out);
    assert(outfile.is_open());

    outfile << A;
}

template <typename T>
Array2D<T> loadArray2D(const std::string &filename)
{
    std::ifstream infile;
    infile.open(filename.c_str(), std::ifstream::in);
    assert(infile.is_open());

    Array2D<T> A;
    infile >> A;

    return A;
}


// B = A(r0:r1, c0:c1)
template <typename T>
Array2D<T> getsub(const Array2D<T> &A, int r0, int r1, int c0, int c1)
{
    assert(r0 >= 0 && c0 >= 0);
    assert(r1 < A.dim1() && c1 < A.dim2());

    int m = r1-r0+1;
    int n = c1-c0+1;
    assert(m > 0 && n > 0);

    int indB = 0;
    int indA = r0+A.dim1()*c0;
    int colstep = r0+A.dim1()-r1-1;
    assert(colstep >= 0);

    Array2D<T> B(m, n);
//     // slower implementation but makes no assumption about storage
//     for(int i = 0; i < m; i++)
//         for(int j = 0; j < n; j++)
//             B(i, j) = A(i+r0, j+c0);
    // faster implementation but assumes column major storage
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            B(indB) = A(indA);
            ++indA;
            ++indB;
        }
        indA += colstep;
    }

    return B;
}

// B = A(r0:r1, :)
template <typename T>
Array2D<T> getrows(const Array2D<T> &A, int r0, int r1)
{
    return getsub(A, r0, r1, 0, A.dim2()-1);
}

// B = A(:, c0:c1)
template <typename T>
Array2D<T> getcols(const Array2D<T> &A, int c0, int c1)
{
    return getsub(A, 0, A.dim1()-1, c0, c1);
}

// B = A(r, :)
template <typename T>
Array2D<T> getrow(const Array2D<T> &A, int r)
{
    return getrows(A, r, r);
}

// B = A(:, c)
template <typename T>
Array2D<T> getcol(const Array2D<T> &A, int c)
{
    return getcols(A, c, c);
}

// A(r0:r1, c0:c1) = B
template <typename T>
void setsub(Array2D<T> &A, int r0, int r1, int c0, int c1, const Array2D<T> &B)
{
    assert(r0 >= 0 && c0 >= 0);
    assert(r1 < A.dim1() && c1 < A.dim2());

    int m = r1-r0+1;
    int n = c1-c0+1;
    assert(m > 0 && n > 0);
    assert(m == B.dim1() && n == B.dim2());

    int indB = 0;
    int indA = r0+A.dim1()*c0;
    int colstep = r0+A.dim1()-r1-1;
    assert(colstep >= 0);

//     // slower implementation but makes no assumption about storage
//     for(int i = 0; i < m; i++)
//         for(int j = 0; j < n; j++)
//             A(i+r0, j+c0) = B(i, j);
    // faster implementation but assumes column major storage
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            A(indA) = B(indB);
            ++indA;
            ++indB;
        }
        indA += colstep;
    }
}

// A(r0:r1, :) = B
template <typename T>
void setrows(Array2D<T> &A, int r0, int r1, const Array2D<T> &B)
{
    return setsub(A, r0, r1, 0, A.dim2()-1, B);
}

// A(:, c0:c1) = B
template <typename T>
void setcols(Array2D<T> &A, int c0, int c1, const Array2D<T> &B)
{
    return setsub(A, 0, A.dim1()-1, c0, c1, B);
}

// A(r, :) = B
template <typename T>
void setrow(Array2D<T> &A, int r, const Array2D<T> &B)
{
    return setrows(A, r, r, B);
}

// A(:, c) = B
template <typename T>
void setcol(Array2D<T> &A, int c, const Array2D<T> &B)
{
    return setcols(A, c, c, B);
}


// B = A'
template <typename T>
Array2D<T> transpose(const Array2D<T> &A)
{
    Array2D<T> B(A.dim2(), A.dim1());
    for(int i = 0; i < A.dim1(); i++)
        for(int j = 0; j < A.dim2(); j++)
            B(j, i) = A(i, j);

    return B;
}

// B = A(end:-1:1, :)
template <typename T>
Array2D<T> flipud(const Array2D<T> &A)
{
    Array2D<T> B(A.dim1(), A.dim2());
    for(int i = 0; i < A.dim1(); i++)
        for(int j = 0; j < A.dim2(); j++)
            B(i, j) = A(A.dim1()-i-1, j);

    return B;
}

// B = A(:, end:-1:1)
template <typename T>
Array2D<T> fliplr(const Array2D<T> &A)
{
    Array2D<T> B(A.dim1(), A.dim2());
    for(int i = 0; i < A.dim1(); i++)
        for(int j = 0; j < A.dim2(); j++)
            B(i, j) = A(i, A.dim2()-j-1);

    return B;
}

// B = repmat(A, m, n);
template <typename T>
Array2D<T> repmat(const Array2D<T> &A, unsigned int m, unsigned int n)
{
    int nrows = A.dim1(), ncols = A.dim2();

    Array2D<T> B(nrows*m, ncols*n);
    for(unsigned int i = 0; i < m; i++)
        for(unsigned int j = 0; j < n; j++)
        {
            unsigned int row_start = i*nrows;
            unsigned int col_start = j*ncols;

            for(int k = 0; k < nrows; k++)
                for(int l = 0; l < ncols; l++)
                    B(row_start+k, col_start+l) = A(k, l);
        }

    return B;
}

// B = reshape(A, m, n)
template <typename T>
Array2D<T> reshape(const Array2D<T> &A, unsigned int m, unsigned int n)
{
    Array2D<T> B = A;
    B.reshape(m, n);

    return B;
}

// C = [A B]
template <typename T>
Array2D<T> horzcat(const Array2D<T> &A, const Array2D<T> &B)
{
    assert(A.dim1()==B.dim1());
    
    Array2D<T> C(A.dim1(), A.dim2()+B.dim2());
    
    for(int r = 0; r < A.dim1(); r++)
    {
        for(int c = 0; c < A.dim2(); c++)
        {
            C(r, c) = A(r, c);
        }
    }

    for(int r = 0; r < B.dim1(); r++)
    {
        for(int c = 0; c < B.dim2(); c++)
        {
            C(r, A.dim2()+c) = B(r, c);
        }
    }
    
    return C;
}

// C = [A; B]
template <typename T>
Array2D<T> vertcat(const Array2D<T> &A, const Array2D<T> &B)
{
    assert(A.dim2()==B.dim2());
    
    Array2D<T> C(A.dim1()+B.dim1(), A.dim2());
    
    for(int r = 0; r < A.dim1(); r++)
    {
        for(int c = 0; c < A.dim2(); c++)
        {
            C(r, c) = A(r, c);
        }
    }

    for(int r = 0; r < B.dim1(); r++)
    {
        for(int c = 0; c < B.dim2(); c++)
        {
            C(A.dim1()+r, c) = B(r, c);
        }
    }
    
    return C;
}


// a = diag(A)
template <typename T>
Array2D<T> diag(const Array2D<T> &A)
{
    if(A.dim1() == 1 || A.dim2() == 1)
    {
        int n = A.size();

        Array2D<T> res(n, n, T(0));
        for(int i = 0; i < n; i++)
            res(i, i) = A(i);

        return res;
    }
    else
    {
        int mn = std::min(A.dim1(), A.dim2());

        Array2D<T> res(mn, 1);
        for(int i = 0; i < mn; i++)
            res(i) = A(i, i);

        return res;
    }
}


// convert
template <typename T, typename S> 
void convert(Array2D<T> &B, const Array2D<S> &A)
{
    assert(B.dim1() == A.dim1());
    assert(B.dim2() == A.dim2());

    for(int i = 0; i < A.size(); i++)
        B(i) = static_cast<T>(A(i));
}



//////////////////////////////////////////////////////////////////////////////
// additional functions and operators
//////////////////////////////////////////////////////////////////////////////

// B = A > s
template <typename T>
Array2D<bool> operator>(const Array2D<T> &A, const T &s)
{
    Array2D<bool> B(A.dim1(), A.dim2());
    greaterThan(B.vector(), A.vector(), s);

    return B;
}

// A = eye(dim)
template <typename T>
Array2D<T> eye(int dim)
{
    Array2D<T> A(dim, dim, T(0));
    for(int i = 0; i < dim; i++)
        A(i, i) = T(1);

    return A;
}

// B = zeros(dim1, dim2)
template <typename T>
Array2D<T> zeros(int dim1, int dim2)
{
    return Array2D<T>(dim1, dim2, T(0));
}

// B = zeros(dim)
template <typename T>
Array2D<T> zeros(int dim)
{
    return zeros<T>(dim, dim);
}

// B = ones(dim1, dim2)
template <typename T>
Array2D<T> ones(int dim1, int dim2)
{
    return Array2D<T>(dim1, dim2, T(1));
}

// B = zeros(dim)
template <typename T>
Array2D<T> ones(int dim)
{
    return ones<T>(dim, dim);
}

// test square matrix
template <typename T>
bool square(const Array2D<T> &A)
{
    return A.dim1() == A.dim2();
}



//////////////////////////////////////////////////////////////////////////////
// matrix elementwise operators
//////////////////////////////////////////////////////////////////////////////

// C = A + B
template <typename T>
Array2D<T> operator+(const Array2D<T> &A, const Array2D<T> &B)
{
    assert(A.dim1() == B.dim1());
    assert(A.dim2() == B.dim2());

    Array2D<T> C(A.dim1(), A.dim2());
    add(C.vector(), A.vector(), B.vector());

    return C;
}

// C = A - B
template <typename T>
Array2D<T> operator-(const Array2D<T> &A, const Array2D<T> &B)
{
    assert(A.dim1() == B.dim1());
    assert(A.dim2() == B.dim2());

    Array2D<T> C(A.dim1(), A.dim2());
    sub(C.vector(), A.vector(), B.vector());

	return C;
}

// C = A .* B
template <typename T>
Array2D<T> dotmult(const Array2D<T> &A, const Array2D<T> &B)
{
	assert(A.dim1() == B.dim1());
    assert(A.dim2() == B.dim2());

	Array2D<T> C(A.dim1(), A.dim2());
	dotmult(C.vector(), A.vector(), B.vector());

	return C;
}

// C = A ./ B
template <typename T>
Array2D<T> dotdiv(const Array2D<T> &A, const Array2D<T> &B)
{
	assert(A.dim1() == B.dim1());
    assert(A.dim2() == B.dim2());

	Array2D<T> C(A.dim1(), A.dim2());
	dotdiv(C.vector(), A.vector(), B.vector());

	return C;
}

// A = A + B
template <typename T>
Array2D<T>& operator+=(Array2D<T> &A, const Array2D<T> &B)
{
	assert(A.dim1() == B.dim1());
    assert(A.dim2() == B.dim2());

    add(A.vector(), A.vector(), B.vector());

    return A;
}

// A = A | B
template <typename T>
Array2D<T> dotor(const Array2D<T> &A, const Array2D<T> &B)
{
	assert(A.dim1() == B.dim1());
    assert(A.dim2() == B.dim2());

    Array2D<T> C(A.dim1(), A.dim2());
    for(int i = 0; i < A.size(); i++)
        C(i) = A(i)|B(i);

    return C;
}



//////////////////////////////////////////////////////////////////////////////
// matrix matrix operators
//////////////////////////////////////////////////////////////////////////////

// C = A * B
template <typename T>
Array2D<T> operator*(const Array2D<T> &A, const Array2D<T> &B)
// Array2D<T> matmult(const Array2D<T> &A, const Array2D<T> &B)
{
    assert(A.dim2() == B.dim1());

    int M = A.dim1();
    int N = A.dim2();
    int K = B.dim2();

    Array2D<T> C(M, K);
    for(int i = 0; i < M; i++)
        for(int j = 0; j < K; j++)
        {
            T sum = 0;

            for(int k = 0; k < N; k++)
                sum += A(i, k)*B(k, j);

            C(i, j) = sum;
        }

    return C;
}



//////////////////////////////////////////////////////////////////////////////
// matrix vector operators
//////////////////////////////////////////////////////////////////////////////

// c = A * b
template <typename T>
Array1D<T> operator*(const Array2D<T> &A, const Array1D<T> &b)
{
    assert(A.dim2() == b.dim());

    int n1 = A.dim1(), n2 = A.dim2();

    Array1D<T> c(n1, 0.0);
    for(int i = 0; i < n1; i++)
        for(int j = 0; j < n2; j++)
            c[i] += A(i, j) * b[j];

    return c;
}

// c = a * B
template <typename T>
Array1D<T> operator*(const Array1D<T> &a, const Array2D<T> &B)
{
    assert(a.dim() == B.dim1());

    int n1 = B.dim1(), n2 = B.dim2();

    Array1D<T> c(n2, 0.0);
    for(int i = 0; i < n2; i++)
        for(int j = 0; j < n1; j++)
            c[i] += a[j] * B(j, i);

    return c;
}



//////////////////////////////////////////////////////////////////////////////
// matrix scalar operators
//////////////////////////////////////////////////////////////////////////////

// B = -A
template <typename T>
Array2D<T> operator-(const Array2D<T> &A)
{
    Array2D<T> B(A.dim1(), A.dim2());
    negate(B.vector(), A.vector());

    return B;
}

// B = A + s
template <typename T>
Array2D<T> operator+(const Array2D<T> &A, const T &s)
{
    Array2D<T> B(A.dim1(), A.dim2());
    add(B.vector(), A.vector(), s);

    return B;
}

// B = A - s
template <typename T>
Array2D<T> operator-(const Array2D<T> &A, const T &s)
{
    Array2D<T> B(A.dim1(), A.dim2());
    sub(B.vector(), A.vector(), s);

    return B;
}

// B = s - A
template <typename T>
Array2D<T> operator-(const T &s, const Array2D<T> &A)
{
    Array2D<T> B(A.dim1(), A.dim2());
    sub(B.vector(), s, A.vector());

    return B;
}

// B = A * s
template <typename T>
Array2D<T> operator*(const Array2D<T> &A, const T &s)
{
    Array2D<T> B(A.dim1(), A.dim2());
    dotmult(B.vector(), A.vector(), s);

    return B;
}

// B = s * A
template <typename T>
Array2D<T> operator*(const T &s, const Array2D<T> &A)
{
    return A*s;
}

// B = A ./ s
template <typename T>
Array2D<T> operator/(const Array2D<T> &A, const T &s)
{
	return A*(1.0/s);
}

// B = s ./ A
template <typename T>
Array2D<T> operator/(const T &s, const Array2D<T> &A)
{
    Array2D<T> B(A.dim1(), A.dim2());
    dotdiv(B.vector(), s, A.vector());

    return B;
}


//////////////////////////////////////////////////////////////////////////////
// trace and more utility functions
//////////////////////////////////////////////////////////////////////////////

// s = trace(A)
template <typename T>
T trace(const Array2D<T> &A)
{
    assert(isSquareMatrix(A));

    int n = A.dim1();

    T res(0);
    for(int i = 0; i < n; i++)
        res += A(i, i);

    return res;
}

// s = trace(A * B)
template <typename T>
T trace_matmult(const Array2D<T> &A, const Array2D<T> &B)
{
    assert(A.dim1() == B.dim2());
    assert(A.dim2() == B.dim1());

    return sum(sum(dotmult(A, transpose(B))))(0, 0);
}


//////////////////////////////////////////////////////////////////////////////
// sum, prod, cumsum, mean and max operators
//////////////////////////////////////////////////////////////////////////////

// b = sum(A(:))
template <typename T>
T sum_vec(const Array2D<T> &A)
{
    const int n = A.dim1()*A.dim2();

    T s(0);
    for(int i = 0; i < n; i++)
      s += A(i);
      
    return s;
}

// B = sum(A, dim)
template <typename T>
Array2D<T> sum(const Array2D<T> &A, const std::string &dim)
{
    if(A.dim1() == 0 && A.dim2() == 0)
        return Array2D<T>();

    Array2D<T> B;
    if(((A.dim1() == 1) || (A.dim2() == 1)) && (dim == ""))
    {
        B = Array2D<T>(1, 1, T(0));
        const int nm = A.dim1()*A.dim2();
        for(int i = 0; i < nm; i++)
            B(0) += A(i);
    }
    else if((dim == "rows") || (dim == ""))
    {
        B = Array2D<T>(1, A.dim2(), T(0));
        for(int i = 0; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                B(j) += A(i, j);
    }
    else if(dim == "cols")
    {
        B = Array2D<T>(A.dim1(), 1, T(0));
        for(int i = 0; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                B(i) += A(i, j);
    }
    else
        assert(false);

    return B;
}

// B = cumsum(A, dim)
template <typename T>
Array2D<T> cumsum(const Array2D<T> &A, const std::string &dim)
{
    if(A.dim1() == 0 && A.dim2() == 0)
        return Array2D<T>();

    Array2D<T> B(A.dim1(), A.dim2());
    if(((A.dim1() == 1) || (A.dim2() == 1)) && (dim == ""))
    {
        B = Array2D<T>(1, 1, T(0));
        const int nm = A.dim1()*A.dim2();
        B(0) = A(0);
        for(int i = 1; i < nm; i++)
            B(i) = B(i-1) + A(i);
    }
    else if((dim == "rows") || (dim == ""))
    {
        for(int j = 0; j < A.dim2(); j++)
            B(0, j) = A(0, j);
        for(int i = 1; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                B(i, j) = B(i-1, j) + A(i, j);
    }
    else if(dim == "cols")
    {
        for(int i = 0; i < A.dim1(); i++)
        {
            B(i, 0) = A(i, 0);
            for(int j = 1; j < A.dim2(); j++)
                B(i, j) = B(i, j-1) + A(i, j);
        }
    }
    else
        assert(false);

    return B;
}

// B = mean(A, dim)
template <typename T>
Array2D<T> mean(const Array2D<T> &A, const std::string &dim)
{
    if(A.dim1() == 0 && A.dim2() == 0)
        return Array2D<T>();

    Array2D<T> B;
    if(((A.dim1() == 1) || (A.dim2() == 1)) && (dim == ""))
    {
        B = Array2D<T>(1, 1, T(0));
        for(int i = 0; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                B(0, 0) += A(i, j);
        B = B/T(std::max(A.dim1(), A.dim2()));
    }
    else if((dim == "rows") || (dim == ""))
    {
        B = Array2D<T>(1, A.dim2(), T(0));
        for(int i = 0; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                B(0, j) += A(i, j);

        B = B/T(A.dim1());
    }
    else if(dim == "cols")
    {
        B = Array2D<T>(A.dim1(), 1, T(0));
        for(int i = 0; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                B(i, 0) += A(i, j);

        B = B/T(A.dim2());
    }
    else
        assert(false);

    return B;
}

// B = max(A, dim)
template <typename T>
Array1D<T> max(const Array2D<T> &A, const std::string &dim)
{
    assert(A.dim1() > 0);
    assert(A.dim2() > 0);

    Array1D<T> B;
    if(((A.dim1() == 1) || (A.dim2() == 1)) && (dim == ""))
    {
        B = Array1D<T>(1, T(0));
        B[0] = A(0, 0);
        for(int i = 0; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                if(A(i, j) > B[0])
                    B[0] = A(i, j);
    }
    else if((dim == "rows") || (dim == ""))
    {
        B = Array1D<T>(A.dim2(), T(0));
        for(int j = 0; j < A.dim2(); j++)
            B[j] = A(0, j);
        for(int i = 1; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                if(A(i, j) > B[j])
                    B[j] = A(i, j);
    }
    else if(dim == "cols")
    {
        B = Array1D<T>(A.dim1(), T(0));
        for(int i = 0; i < A.dim1(); i++)
            B[i] = A(i, 0);
        for(int i = 0; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                if(A(i, j) > B[i])
                    B[i] = A(i, j);
    }
    else
        assert(false);

    return B;
}

// B = min(A, dim)
template <typename T>
Array1D<T> min(const Array2D<T> &A, const std::string &dim)
{
    if(A.dim1() == 0 && A.dim2() == 0)
        return Array1D<T>();

    Array1D<T> B;
    if(((A.dim1() == 1) || (A.dim2() == 1)) && (dim == ""))
    {
        B = Array1D<T>(1, T(0));
        B[0] = A(0, 0);
        for(int i = 0; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                if(A(i, j) < B[0])
                    B[0] = A(i, j);
    }
    else if((dim == "rows") || (dim == ""))
    {
        B = Array1D<T>(A.dim2(), T(0));
        for(int j = 0; j < A.dim2(); j++)
            B[j] = A(0, j);
        for(int i = 1; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                if(A(i, j) < B[j])
                    B[j] = A(i, j);
    }
    else if(dim == "cols")
    {
        B = Array1D<T>(A.dim1(), T(0));
        for(int i = 0; i < A.dim1(); i++)
            B[i] = A(i, 0);
        for(int i = 0; i < A.dim1(); i++)
            for(int j = 0; j < A.dim2(); j++)
                if(A(i, j) < B[i])
                    B[i] = A(i, j);
    }
    else
        assert(false);

    return B;
}



//////////////////////////////////////////////////////////////////////////////
// variance operators
//////////////////////////////////////////////////////////////////////////////

// B = cov(A)
template <typename T>
Array2D<T> cov(const Array2D<T> &A)
{
    if(A.dim1() == 0 && A.dim2() == 0)
        return Array2D<T>();

    Array2D<T> mu = mean(A);
    return (transpose(A)*A)/T(A.dim1()) - transpose(mu)*mu;
}

// B = var(A)
template <typename T>
Array2D<T> var(const Array2D<T> &A)
{
    Array2D<T> mu = mean(A);
    return sum(dotmult(A, A))/T(A.dim1()) - dotmult(mu, mu);
}

// B = cov(a)
template <typename T>
Array2D<T> cov(const Array1D<T> &a)
{
    int n = a.dim();

    Array2D<T> B(n, n);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            B(i, j) = a[i]*a[j];

    return B;
}



//////////////////////////////////////////////////////////////////////////////
// distance operators
//////////////////////////////////////////////////////////////////////////////

// s = b'*A*b
template <typename T>
T mahalanobis(const Array2D<T> &A, const Array1D<T> &b)
{
    return dot(b*A, b);
}

// s = diag(B'*A*B)
template <typename T>
Array2D<T> mahalanobis(const Array2D<T> &A, const Array2D<T> &B)
{
    return sum(dotmult(transpose(B)*A, B), "cols");
}

// C = dist2(A, B);
template <typename T>
Array2D<T> dist2(const Array2D<T> &A, const Array2D<T> &B)
{
    assert(A.dim2() == B.dim2());

    return repmat(sum(dotmult(A, A), "cols"), 1, B.dim1()) +
           repmat(transpose(sum(dotmult(B, B), "cols")), A.dim1(), 1) - 2.0*(A*transpose(B));
}



//////////////////////////////////////////////////////////////////////////////
// overloaded maths operators
//////////////////////////////////////////////////////////////////////////////

// B = log(a)
template <typename T>
Array2D<T> log(const Array2D<T> &A)
{
    int n = A.dim1()*A.dim2();

    Array2D<T> B(A.dim1(), A.dim2());
    for(int i = 0; i < n; i++)
        B(i) = std::log(A(i));

    return B;
}

// B = exp(a)
template <typename T>
Array2D<T> exp(const Array2D<T> &A)
{
    int n = A.dim1()*A.dim2();

    Array2D<T> B(A.dim1(), A.dim2());
    for(int i = 0; i < n; i++)
        B(i) = std::exp(A(i));

    return B;
}

// B = sqrt(A)
template <typename T>
Array2D<T> sqrt(const Array2D<T> &A)
{
    Array2D<T> B(A.dim1(), A.dim2());
    sqrt(B.vector(), A.vector());

    return B;
}

// B = fabs(A)
template <typename T>
Array2D<T> fabs(const Array2D<T> &A)
{
    Array2D<T> B(A.dim1(), A.dim2());
    fabs(B.vector(), A.vector());

    return B;
}

// assert functions
template <typename T> 
bool isSquareMatrix(const Array2D<T> &A)
{
    return A.dim1() == A.dim2();
}

// normalisation functions
template <typename T>
Array2D<T> makeHomogeneous(const Array2D<T> &A)
{
    Array2D<T> B(A);
    for(int r = 0; r < B.dim1(); r++)
    {
        for(int c = 0; c < B.dim2(); c++)
        {
            B(r, c) /= (B(B.dim1()-1, c)+std::numeric_limits<T>::min());
        }
    }
    
    return B;
}

template <typename T>
Array2D<T> normaliseCols(const Array2D<T> &A)
{
    Array2D<T> lens = sqrt(sum(dotmult(A, A)));
    
    Array2D<T> B(A);
    for(int r = 0; r < B.dim1(); r++)
    {
        for(int c = 0; c < B.dim2(); c++)
        {
            B(r, c) /= (lens(c)+std::numeric_limits<T>::min());
        }
    }
    
    return B;
}

// find function
template <typename T, typename UnaryPred>
std::vector<int> find(const Array2D<T> &v, const UnaryPred pred)
{
    std::vector<int> inds;
    for(int i = 0; i < v.size(); i++)
        if(pred(v(i)))
            inds.push_back(i);

    return inds;
}

}

#endif
