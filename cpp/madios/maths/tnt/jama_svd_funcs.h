#ifndef JAMA_SVD_FUNCS_H
#define JAMA_SVD_FUNCS_H

#include "jama_svd.h"


// B = pinv(A)
// B is the premultiply (pseudo) inverse, i.e. B*A == I
template <typename T>
Array2D<T> pinv(const Array2D<T> &A, bool include_nullspace=false)
{
    JAMA::SVD<T> svd(A);
    Array2D<T> U = svd.getU();
    Array2D<T> S = svd.getS();
    Array2D<T> V = svd.getV();
    Array2D<T> s = diag(S);

    int count = 0;
    T tol(std::min(S.dim1(), S.dim2())*2.2204e-16*max(diag(S).vector()));
    for(int i = 0; i < s.size(); i++)
        if(s(i) > tol)
        {
            s(i) = T(1.0)/s(i);
            count++;
        }
        else
            s(i) = 0.0;

    assert(count > 0);
    Array2D<double> inv1 = getcols(V, 0, count-1) * diag(s) * transpose(getcols(U, 0, count-1));
    if(include_nullspace)
    {
        Array2D<double> inv2(inv1.dim1(), S.dim2());
        for(int i = 0; i < inv1.dim1(); i++)
        {
            for(int j = 0; j < inv1.dim2(); j++)
                inv2(i, j) = inv1(i, j);
            for(int j = inv1.dim2(); j < inv2.dim2(); j++)
                inv2(i, j) = V(i, j);
        }

        return inv2;
    }
    else
        return inv1;
}

#endif
