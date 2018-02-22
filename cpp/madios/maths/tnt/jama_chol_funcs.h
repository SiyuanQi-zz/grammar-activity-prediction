#ifndef JAMA_CHOL_FUNCS_H
#define JAMA_CHOL_FUNCS_H

#include "jama_chol.h"


// B = spdinv(A)
// positive definite symmstric matrix inverse
template <typename T>
Array2D<T> spdinv(const Array2D<T> &A)
{
    assert(isSquareMatrix(A));

    JAMA::Cholesky<T> chol(A);
    Array2D<T> B(chol.solve(eye<T>(A.dim1())));
    assert(B.dim1() > 0);

    return B;
}

// s = logdet(A)
// log of determinant of symmetric positive definitive matrix
template <typename T>
T logdet(const Array2D<T> &A)
{
    assert(isSquareMatrix(A));
    
    return static_cast<T>(2)*sum_vec(log(diag(JAMA::Cholesky<T>(A).getL())+std::numeric_limits<T>::min()));
}

#endif
