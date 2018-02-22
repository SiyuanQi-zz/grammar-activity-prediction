#ifndef JAMA_LU_FUNCS_H
#define JAMA_LU_FUNCS_H

#include "jama_lu.h"

using TNT::eye;


// B = inv(A)
// matrix inverse
template <typename T>
Array2D<T> inv(const Array2D<T> &A)
{
    assert(square(A));

    JAMA::LU<T> lu(A);
    if(!lu.isNonsingular())
    {
        std::cout << "Nonsingular: " << A << std::endl;
        assert(false);
    }
    Array2D<T> B(lu.solve(eye<T>(A.dim1())));
    assert(B.dim1() > 0);

    return B;
}

// B = inv(A)
// matrix inverse and determinant
template <typename T>
Array2D<T> inv(const Array2D<T> &A, T &det)
{
    assert(square(A));

	JAMA::LU<T> lu(A);
    det = lu.det();
    if(!lu.isNonsingular())
    {
        std::cout << "Nonsingular: " << A << std::endl;
        assert(false);
    }
    Array2D<T> B(lu.solve(eye<T>(A.dim1())));
    assert(B.dim1() > 0);

	return B;
}

// s = det(A)
// matrix determinant
template <typename T>
T det(const Array2D<T> &A)
{
    assert(isSquareMatrix(A));

    JAMA::LU<T> lu(A);
    return lu.det();
}

// C = mldivide(A, B)
// equivalent to C = A \ B
template <typename T>
Array2D<T> mldivide(const Array2D<T> &A, const Array2D<T> &B)
{
    assert(A.dim1() == B.dim1());

    JAMA::LU<T> lu(A);

    return lu.solve(B);
}

// C = mrdivide(A, B) = mldivide(B', A')
// equivalent to C = A / B
template <typename T>
Array2D<T> mrdivide(const Array2D<T> &A, const Array2D<T> &B)
{
    return mldivide(transpose(B), transpose(A));
}

#endif
