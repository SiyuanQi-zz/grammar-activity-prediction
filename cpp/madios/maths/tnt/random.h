#ifndef RAND_H
#define RAND_H

#include "array2d.h"
#include "special.h"


// B = rand(dim1, dim2)
template <typename T>
Array2D<T> rand(int dim1, int dim2)
{
    Array2D<T> B(dim1, dim2);

    int n = B.size();
    for(int i = 0; i < n; i++)
        B(i) = uniform_rand();

    return B;
}

// B = rand(dim)
template <typename T>
Array2D<T> rand(int dim)
{
    return rand<T>(dim, dim);
}

// B = randn(dim1, dim2)
template <typename T>
Array2D<T> randn(int dim1, int dim2)
{
    Array2D<T> B(dim1, dim2);

    int n = B.size();
    for(int i = 0; i < n; i++)
        B(i) = normal_rand();

    return B;
}

// B = randn(dim)
template <typename T>
Array2D<T> randn(int dim)
{
    return randn<T>(dim, dim);
}


#endif
