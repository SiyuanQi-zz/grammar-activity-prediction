#ifndef PROBABILITY_DISTRIBUTIONS
#define PROBABILITY_DISTRIBUTIONS

#include "tnt/array2d.h"
#include "tnt/jama_eig.h"
#include "tnt/random.h"
#include "tnt/jama_lu.h"
#include "tnt/jama_lu_funcs.h"
#include "STLArithmetics.h"
#include "special.h"
#include "Constants.h"

const double pi = MathConstants::PI;

template <typename T>
T gammaln(const T &nu, unsigned int N)
{
    std::vector<T> args(N);
    iota(args.begin(), args.end(), nu-N+1);

    T res = (N*(N-1)*0.25)*log(pi);
    for(unsigned int i = 0; i < args.size(); i++)
        res += gammaln(args[i]*0.5);

    return res;
}


// Normal Distribution

template <typename T>
Array2D<T> gaussln(const Array2D<T> &mu, const Array2D<T> &covar, const Array2D<T> &X)
{
    assert(mu.dim2() == covar.dim1());
    assert(isSquareMatrix(covar));

    int n = X.dim1();
    int d = X.dim2();

    Array2D<T> inv_covar = inv(covar);
    Array2D<T> X_mu = X - ones<T>(n, 1)*reshape(mu, 1, d);
    Array2D<T> fact = sum(dotmult(X_mu*inv_covar, X_mu), "cols");

    Array2D<T> y = static_cast<T>(-0.5*(d*log(2.0*pi) + logdet(covar))) - static_cast<T>(0.5)*fact;

    return y;
}

template <typename T>
Array2D<T> gauss(const Array2D<T> &mu, const Array2D<T> &covar, const Array2D<T> &X)
{
    return exp(gaussln(mu, covar, X));
}

template <typename T>
Array2D<T> gsamp(const Array2D<T> &mu, const Array2D<T> &covar, int nsamples)
{
    assert(mu.dim2() == covar.dim1());
    assert(isSquareMatrix(covar));

    int d = mu.dim2();

    JAMA::Eigenvalue<T> eig(covar);
    Array2D<T> evec = eig.getV();
    Array2D<T> eval = eig.getD();

    Array2D<T> coeffs = randn<T>(nsamples, d)*sqrt(eval);
    Array2D<T> X = ones<T>(nsamples, 1)*mu + coeffs*transpose(evec);

    return X;
}


// Wishart Distribution

template <typename T>
T wishart_exp_logdet(const Array2D<T> &W, const T &v)
{
    int D = W.dim1();

    T res = D*log(2) + logdet(W);
    for(int i = 0; i < D; i++)
        res += digamma(0.5*(v-i));

    return res;
}

template <typename T>
T wishartln_denom(const Array2D<T> &W, const T &v)
{
    assert(isSquareMatrix(W));

    int D = W.dim1();

    return -((0.5*v*D)*log(2) + gammaln(v, D) + 0.5*v*logdet(W));
}

template <typename T>
T wishart_entropy(const Array2D<T> &W, const T &v)
{
    assert(isSquareMatrix(W));

    int D = W.dim1();

    return -wishartln_denom(W, v) - 0.5*(v-D-1)*wishart_exp_logdet(W, v) + 0.5*v*D;
}


// compute normalisation constant
template <typename T>
T computeNormalisationConstant(const std::vector<T> &ss)
{
    // estimate log(k) by binary search
    int n = ss.size();
    T min_logk = log(std::max(-log(2.0/n), realmin))-log(max(ss)+realmin);
    T max_logk = log(std::max(-log(0.5/n), realmin))-log(min(ss)+realmin);
    T min_logk_Z = computeZ(min_logk, ss);
    T max_logk_Z = computeZ(max_logk, ss);
    assert(min_logk_Z >= 1);
    assert(max_logk_Z <= 1);

    int count = 0;
    T logk = 0.5*(min_logk+max_logk);
    T logk_Z = computeZ(logk, ss);
    while(fabs(logk_Z - 1) > 1e-5)
    {
        count = count + 1;
        if(logk_Z < 1)
            max_logk = logk;
        else
            min_logk = logk;

        logk = 0.5*(min_logk+max_logk);
        logk_Z = computeZ(logk, ss);
    }

    T k = exp(logk);

    return k;
}

template <typename T>
std::vector<T> computeRD(const T &logk, const std::vector<T> &ss)
{
    return exp(-exp(logk)*ss);
}

template <typename T>
T computeZ(const T &logk, const std::vector<T> &ss)
{
    std::vector<T> R = computeRD(logk, ss);

    return sum(R);
}

#endif
