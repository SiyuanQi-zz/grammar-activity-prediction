#ifndef SPECIAL_H
#define SPECIAL_H

#include <cmath>
#include <vector>
#include <limits>

extern const double realmin;
extern const double realmax;
extern const int intmax;

double uniform_rand();
double uniform_rand(double l, double u);
double normal_rand();
double normal_rand(double mu, double stddev);

double cot(double x);

double gammaln(double x);
double digamma(double x);

double factln(unsigned int n);
double binom(unsigned int k, unsigned int n, double p);

//return MAX number of distinct real roots
unsigned int solve_cubic(double a, double b, double c, double d, double &result0, double &result1, double &result2);

template <class T>
inline void clamp(const T& min, const T& max, T& v)
{
    if (v < min) v = min;
    if (v > max) v = max;
}

#endif
