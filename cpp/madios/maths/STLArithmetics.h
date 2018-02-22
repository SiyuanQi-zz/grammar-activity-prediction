#ifndef STL_ARITHMETICS_H
#define STL_ARITHMETICS_H

#include "special.h"

#include <numeric>
#include <algorithm>


// utility functions

template <typename T>
std::vector<T> reverse(const std::vector<T> &seq)
{
    std::vector<T> qes = seq;
    reverse(qes.begin(), qes.end());
    return qes;
}

template <typename T>
std::vector<T> sort(const std::vector<T> &seq)
{
    std::vector<T> eqs = seq;
    sort(eqs.begin(), eqs.end());
    return eqs;
}

template <typename T>
std::vector<T> unique(const std::vector<T> &seq)
{
    std::vector<T> seq2 = seq;
//    sort(eqs.begin(), eqs.end());
    seq2.erase(std::unique(seq2.begin(), seq2.end()), seq2.end());
    return seq2;
}

template <typename T>
std::vector<int> count(const std::vector<T> &seq, const std::vector<T> &words)
{
    std::vector<int> counts(words.size(), 0);
    for(int i = 0; i < static_cast<int>(seq.size()); i++)
        for(int j = 0; j < static_cast<int>(words.size()); j++)
            if(seq[i] == words[j])
                counts[j]++;
    return counts;
}

template <typename T>
std::vector<double> sum(const std::vector<T> &seq, const std::vector<double> &w, const std::vector<T> &words)
{
    assert(seq.size() == w.size());

    std::vector<double> counts(words.size(), 0);
    for(int i = 0; i < static_cast<int>(seq.size()); i++)
        for(int j = 0; j < static_cast<int>(words.size()); j++)
            if(seq[i] == words[j])
                counts[j] += w[i];
    return counts;
}


// arithmetic functions for STL vectors

template <typename T>
std::vector<T> operator+(const std::vector<T> &v1, const std::vector<T> &v2)
{
    assert(v1.size() == v2.size());

    std::vector<T> res(v1);
    for(unsigned int i = 0; i < v1.size(); i++)
        res[i] += v2[i];

    return res;
}

template <typename T>
std::vector<T>& operator+=(std::vector<T> &v1, const std::vector<T> &v2)
{
    assert(v1.size() == v2.size());

    for(unsigned int i = 0; i < v1.size(); i++)
        v1[i] += v2[i];

    return v1;
}

template <typename T>
std::vector<T> operator*(const std::vector<T> &v1, const std::vector<T> &v2)
{
    std::vector<T> res(v1.size());
    transform(v1.begin(), v1.end(), v2.begin(), res.begin(), std::multiplies<T>());
    return res;
}

template <typename T>
std::vector<T>& operator*=(std::vector<T> &v1, const std::vector<T> &v2)
{
    assert(v1.size() == v2.size());

    for(unsigned int i = 0; i < v1.size(); i++)
        v1[i] *= v2[i];

    return v1;
}

template <typename T, typename S>
std::vector<T> operator+(const std::vector<T> &v, const S &s)
{
    std::vector<T> res(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        res[i] = v[i]+s;
    return res;
}

template <typename S, typename T>
std::vector<T> operator+(const S &s, const std::vector<T> &v)
{
    std::vector<T> res(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        res[i] = v[i]+s;
    return res;
}

template <typename T>
std::vector<T> operator-(const std::vector<T> &v, const std::vector<T> &w)
{
    std::vector<T> res(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        res[i] = v[i]-w[i];
    return res;
}

template <typename T, typename S>
std::vector<T> operator-(const std::vector<T> &v, const S &s)
{
    std::vector<T> res(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        res[i] = v[i]-s;
    return res;
}

template <typename T, typename S>
std::vector<T> operator-(const S &s, const std::vector<T> &v)
{
    std::vector<T> res(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        res[i] = s-v[i];
    return res;
}

template <typename T, typename S>
std::vector<T> operator*(const std::vector<T> &v, const S &s)
{
    std::vector<T> res(v);
    for(unsigned int i = 0; i < v.size(); i++)
        res[i] *= s;
    return res;
}

template <typename T, typename S>
std::vector<T> operator*(const S &s, const std::vector<T> &v)
{
    return v*s;
}

template <typename T, typename S>
std::vector<T> operator/(const std::vector<T> &v, const S &s)
{
    S den = 1.0/s;

    std::vector<T> res(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        res[i] = v[i]*den;
    return res;
}

template <typename T, typename S>
std::vector<T> operator/(const S &s, const std::vector<T> &v)
{
    std::vector<T> res(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        res[i] = s/v[i];
    return res;
}

template <typename T>
std::vector<T> operator*(const T &s, const std::vector<T> &v)
{
    return v*s;
}

template <typename T>
std::vector<T>& operator*=(std::vector<T> &v, const T &s)
{
    for(unsigned int i = 0; i < v.size(); i++)
        v[i] *= s;
    return v;
}

template <typename T>
std::vector<T> operator/(const std::vector<T> &v, const T &s)
{
    return v*(1.0/s);
}

template <typename T>
std::vector<T>& operator/=(std::vector<T> &v, const T &s)
{
    return v*=(1.0/s);
}

template <typename T>
std::vector<int> dot_le(const std::vector<T> &v, const std::vector<T> &u)
{
    assert(v.size() == u.size());

    std::vector<int> w(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        w[i] = v[i]<u[i];

    return w;
}

template <typename T>
std::vector<int> dot_geq(const std::vector<T> &v, const std::vector<T> &u)
{
    assert(v.size() == u.size());

    std::vector<int> w(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        w[i] = v[i]>=u[i];

    return w;
}

template <typename T>
std::vector<T> dot_or(const std::vector<T> &v, const std::vector<T> &u)
{
    assert(v.size() == u.size());

    std::vector<T> w(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        w[i] = v[i] | u[i];

    return w;
}

template <typename T>
std::vector<T> dot_and(const std::vector<T> &v, const std::vector<T> &u)
{
    assert(v.size() == u.size());

    std::vector<T> w(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        w[i] = v[i] & u[i];

    return w;
}

template <typename T>
std::vector<T> dotmult(const std::vector<T> &v, const std::vector<T> &u)
{
    assert(v.size() == u.size());

    std::vector<T> w(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
        w[i] = v[i]*u[i];

    return w;
}

template <typename T>
std::vector<T> dotmult(const std::vector<T> &v)
{
    return dotmult(v, v);
}


// special maths functions for STL vector

template <typename T>
std::vector<T> pow(const std::vector<T> &a, const T &c)
{
    std::vector<T> b(a.size());
    for(unsigned int i = 0; i < a.size(); i++)
        b[i] = pow(a[i], c);

    return b;
}

template <typename T>
std::vector<T> exp(const std::vector<T> &a)
{
    std::vector<T> b(a.size());
    for(unsigned int i = 0; i < a.size(); i++)
        b[i] = exp(a[i]);

    return b;
}

template <typename T>
std::vector<T> log(const std::vector<T> &a)
{
    std::vector<T> b(a.size());
    for(unsigned int i = 0; i < a.size(); i++)
        b[i] = log(a[i]);

    return b;
}

template <typename T>
std::vector<T> sqrt(const std::vector<T> &a)
{
    std::vector<T> b(a.size());
    for(unsigned int i = 0; i < a.size(); i++)
        b[i] = sqrt(a[i]);

    return b;
}

template <typename T>
std::vector<T> gammaln(const std::vector<T> &a)
{
    std::vector<T> b(a.size());
    for(unsigned int i = 0; i < a.size(); i++)
        b[i] = gammaln(a[i]);

    return b;
}

template <typename T>
int min_index(const std::vector<T> &w)
{
    assert(w.size() > 0);

    int min_ind = 0;
    for(unsigned int i = 1; i < w.size(); i++)
        if(w[i] < w[min_ind])
            min_ind = i;

    return min_ind;
}

template <typename T>
int max_index(const std::vector<T> &w)
{
    assert(w.size() > 0);

    int max_ind = 0;
    for(unsigned int i = 1; i < w.size(); i++)
        if(w[i] > w[max_ind])
            max_ind = i;

    return max_ind;
}

template <typename T>
const T& min(const std::vector<T> &w)
{
    return w[min_index(w)];
}

template <typename T>
const T& max(const std::vector<T> &w)
{
    return w[max_index(w)];
}

template <typename T>
T sum(const std::vector<T> &w)
{
    return std::accumulate(w.begin(), w.end(), T(0));
}

template <typename T>
std::vector<T> cumsum(const std::vector<T> &w)
{
    if(w.size() == 0)
        return w;

    std::vector<T> res(w.size(), w[0]);
    for(unsigned int i = 1; i < w.size(); i++)
        res[i] = res[i-1]+w[i];

    return res;
}

template <typename T>
T prod(const std::vector<T> &w)
{
    T res(1);
    for(unsigned int i = 0; i < w.size(); i++)
        res *= w[i];

    return res;
}

template <typename T>
std::vector<T> normalise(const std::vector<T> &w)
{
    return w/sum(w);
}

template <typename T>
double mean(const std::vector<T> &v)
{
    double mu = sum(v) / static_cast<double>(v.size());

    return mu;
}

template <typename T>
double mean(const std::vector<T> &v, const std::vector<double> &w)
{
    assert(v.size() == w.size());

    double ws = 0.0;
    double vs = 0.0;
    for(unsigned int i = 0; i < v.size(); i++)
    {
        vs += v[i]*w[i];
        ws += w[i];
    }

    return vs/ws;
}

template <typename T>
double var(const std::vector<T> &v)
{
    double mu = mean(v);

    double sigma2 = 0.0;
    for(unsigned int i = 0; i < v.size(); i++)
        sigma2 += (v[i]-mu)*(v[i]-mu);
    sigma2 = sigma2 / static_cast<double>(v.size());

    return sigma2;
}

template <typename T>
double var(const std::vector<T> &v, const std::vector<double> &w)
{
    assert(v.size() == w.size());

    double mu = mean(v, w);

    double ws = 0.0;
    double sigma2 = 0.0;
    for(unsigned int i = 0; i < v.size(); i++)
    {
        sigma2 += (v[i]-mu)*(v[i]-mu)*w[i];
        ws += w[i];
    }
    sigma2 = sigma2 / ws;

    return sigma2;
}

template <typename T>
double entropy(const std::vector<T> &p)
{
    const double realmin = std::numeric_limits<double>::min();

    return -sum(p*log(p+realmin));
}

// template <typename T>
// double kl(const std::vector<T> &p, const std::vector<T> &q)
// {
//     assert(p.size() == q.size());
//
//     const double realmin = std::numeric_limits<double>::min();
//
//     double d = 0.0;
//     for(unsigned int i = 0; i < p.size(); i++)
//         d += p[i]*(log(p[i]+realmin) - log(q[i]+realmin));
//
//     return d;
// }

template <typename T>
double kl(const std::vector<T> &p, const std::vector<T> &q)
{
    assert(p.size() == q.size());

    const double realmin = std::numeric_limits<double>::min();

    double d = sum(p*(log(p+realmin)-log(q+realmin)));

    return d;
}

template <typename T>
double ssd(const std::vector<T> &p, const std::vector<T> &q)
{
    assert(p.size() == q.size());

    double d = 0.0;
    for(unsigned int i = 0; i < p.size(); i++)
        d += (p[i]-q[i])*(p[i]-q[i]);

    return d;
}

template <typename T>
double dot(const std::vector<T> &p, const std::vector<T> &q)
{
    assert(p.size() == q.size());

    double d = 0.0;
    for(unsigned int i = 0; i < p.size(); i++)
        d += p[i]*q[i];

    return d;
}


// utility functions for STL vector
template <typename T, typename UnaryPred>
std::vector<int> apply_pred(const std::vector<T> &v, const UnaryPred &pred)
{
    std::vector<int> logs(v.size());
    for(unsigned int i= 0; i < v.size(); i++)
    {
        logs[i] = pred(v[i]);
    }

    return logs;
}

template <typename T, typename UnaryPred>
std::vector<int> find(const std::vector<T> &v, const UnaryPred &pred)
{
    std::vector<int> inds;
    for(unsigned int i = 0; i < v.size(); i++)
        if(pred(v[i]))
            inds.push_back(i);

    return inds;
}

template <typename T>
std::vector<int> find(const std::vector<T> &v)
{
    std::vector<int> inds;
    for(unsigned int i = 0; i < v.size(); i++)
        if(v[i] != 0)
            inds.push_back(i);
    return inds;
}

// v[inds]
template <typename T>
std::vector<T> select_index(const std::vector<T> &v, const std::vector<int> &inds)
{
    std::vector<T> res(inds.size());
    for(unsigned int i = 0; i < inds.size(); i++)
        res[i] = v[inds[i]];
    return res;
}

// v[inds]
template <typename T>
std::vector<T> select_logical(const std::vector<T> &v, const std::vector<int> &logs)
{
    assert(v.size()==logs.size());

    std::vector<T> res;
    for(unsigned int i = 0; i < v.size(); i++)
        if(logs[i])
            res.push_back(v[i]);
    return res;
}

// tgt[inds] = src
template <typename T>
void assign(std::vector<T> &tgt, const std::vector<int> &inds, const std::vector<T> &src)
{
    for(unsigned int i = 0; i < inds.size(); i++)
        tgt[inds[i]] = src[i];
}

// tgt[inds] = src
template <typename T>
void assign(std::vector<T> &tgt, const std::vector<int> &inds, const T &src)
{
    for(unsigned int i = 0; i < inds.size(); i++)
        tgt[inds[i]] = src;
}



// Dirichlet Distribution

template <typename T>
T dirichletln(const std::vector<T> &x, const std::vector<T> &a, const std::string &x_type)
{
    assert(x.size() == a.size());

    T const_norm = gammaln(sum(a)) - sum(gammaln(a));
    if(x_type == "log")
        return sum((a-T(1))*x) + const_norm;
    else
        return sum((a-T(1))*log(x)) + const_norm;
}

template <typename T>
std::vector<T> dirichletln_exp(const std::vector<T> &a)
{
    std::vector<T> res(a.size());
    transform(a.begin(), a.end(), res.begin(), digamma);

    return res - digamma(sum(a));
}

#endif
