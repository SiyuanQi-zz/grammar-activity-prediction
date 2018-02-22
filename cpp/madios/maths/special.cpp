#include "special.h"

#include "Constants.h"

using MathConstants::DOUBLE_EPSILON;
using MathConstants::EPSILON;
using MathConstants::PI;

#include <cassert>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>

using std::swap;
using std::vector;


const double INV_RAND_MAX = 1.0 / static_cast<double>(RAND_MAX);
const double realmin = std::numeric_limits<double>::min();
const double realmax = std::numeric_limits<double>::max();
const int intmax = std::numeric_limits<int>::max();


#ifndef cbrt
template <typename T>
T cbrt(const T &v)
{
	return pow(v, 1.0/3.0);
}
#endif

double uniform_rand()
{
    return rand() * INV_RAND_MAX;
}

double uniform_rand(double l, double u)
{
    assert(l < u);
    return rand()*(u-l) + l;
}

double normal_rand()
{
    static double n2 = 0.0;
    static bool cached = false;

    if(cached)
    {
        cached = false;
        return n2;
    }
    else
    {
        double x1, x2, w, n1;

        do
        {
            x1 = 2.0 * uniform_rand() - 1.0;
            x2 = 2.0 * uniform_rand() - 1.0;
            w = x1*x1 + x2*x2;
        }
        while(w >= 1.0);

        w = sqrt((-2.0*log(w)) / w);
        n1 = x1*w;
        n2 = x2*w;

        cached = true;
        return n1;
    }

    assert(false);
    return 0.0;
}

double normal_rand(double mu, double stddev)
{
    return normal_rand()*stddev + mu;
}

double cot(double x)
{
    return 1.0/tan(x);
}

double gammaln(double x)
{
    const double d1 = -5.772156649015328605195174e-1;
    const double p1[8] = {4.945235359296727046734888e0, 2.018112620856775083915565e2,
                          2.290838373831346393026739e3, 1.131967205903380828685045e4,
                          2.855724635671635335736389e4, 3.848496228443793359990269e4,
                          2.637748787624195437963534e4, 7.225813979700288197698961e3};
    const double q1[8] = {6.748212550303777196073036e1, 1.113332393857199323513008e3,
                          7.738757056935398733233834e3, 2.763987074403340708898585e4,
                          5.499310206226157329794414e4, 6.161122180066002127833352e4,
                          3.635127591501940507276287e4, 8.785536302431013170870835e3};
    const double d2 = 4.227843350984671393993777e-1;
    const double p2[8] = {4.974607845568932035012064e0, 5.424138599891070494101986e2,
                          1.550693864978364947665077e4, 1.847932904445632425417223e5,
                          1.088204769468828767498470e6, 3.338152967987029735917223e6,
                          5.106661678927352456275255e6, 3.074109054850539556250927e6};
    const double q2[8] = {1.830328399370592604055942e2, 7.765049321445005871323047e3,
                          1.331903827966074194402448e5, 1.136705821321969608938755e6,
                          5.267964117437946917577538e6, 1.346701454311101692290052e7,
                          1.782736530353274213975932e7, 9.533095591844353613395747e6};
    const double d4 = 1.791759469228055000094023e0;
    const double p4[8] = {1.474502166059939948905062e4, 2.426813369486704502836312e6,
                          1.214755574045093227939592e8, 2.663432449630976949898078e9,
                          2.940378956634553899906876e10, 1.702665737765398868392998e11,
                          4.926125793377430887588120e11, 5.606251856223951465078242e11};
    const double q4[8] = {2.690530175870899333379843e3, 6.393885654300092398984238e5,
                          4.135599930241388052042842e7, 1.120872109616147941376570e9,
                          1.488613728678813811542398e10, 1.016803586272438228077304e11,
                          3.417476345507377132798597e11, 4.463158187419713286462081e11};
    const double c[7] = {-1.910444077728e-03, 8.4171387781295e-04,
                         -5.952379913043012e-04, 7.93650793500350248e-04,
                         -2.777777777777681622553e-03, 8.333333333333333331554247e-02,
                          5.7083835261e-03};

    assert(x > 0.0);

    //0 <= x <= eps
    if (x < DOUBLE_EPSILON)
        return -log(x);

    // eps < x <= 0.5
    if ((x > DOUBLE_EPSILON) && (x <= 0.5))
    {
        double y = x;
        double xden = 1;
        double xnum = 0;
        for (unsigned int i = 0; i < 8; i++)
        {
            xnum = xnum * y + p1[i];
            xden = xden * y + q1[i];
        }

        return -log(y) + (y * (d1 + y * (xnum / xden)));
    }

    //0.5 < x <= 0.6796875
    if ((x > 0.5) && (x <= 0.6796875))
    {
        double xm1 = (x - 0.5) - 0.5;
        double xden = 1;
        double xnum = 0;
        for (unsigned int i = 0; i < 8; i++)
        {
            xnum = xnum * xm1 + p2[i];
            xden = xden * xm1 + q2[i];
        }

        return -log(x) + xm1 * (d2 + xm1 * (xnum / xden));
    }


    //0.6796875 < x <= 1.5
    if ((x > 0.6796875) && (x <= 1.5))
    {
        double xm1 = (x - 0.5) - 0.5;
        double xden = 1;
        double xnum = 0;
        for (unsigned int i = 0; i < 8; i++)
        {
            xnum = xnum * xm1 + p1[i];
            xden = xden * xm1 + q1[i];
        }

        return xm1 * (d1 + xm1 * (xnum / xden));
    }

    //1.5 < x <= 4
    if ((x > 1.5) && (x <= 4.0))
    {
        double xm2 = x - 2;
        double xden = 1;
        double xnum = 0;
        for (unsigned int i = 0; i < 8; i++)
        {
            xnum = xnum * xm2 + p2[i];
            xden = xden * xm2 + q2[i];
        }

        return xm2 * (d2 + xm2 * (xnum / xden));
    }

    //4 < x <= 12
    if ((x > 4.0) & (x <= 12.0))
    {
        double xm4 = x - 4;
        double xden = -1;
        double xnum = 0;
        for (unsigned int i = 0; i < 8; i++)
        {
            xnum = xnum * xm4 + p4[i];
            xden = xden * xm4 + q4[i];
        }

        return d4 + xm4 * (xnum / xden);
    }

    //x > 12
    if (x > 12.0)
    {
        double y = x;
        double r = c[6];
        double ysq = y * y;
        for (unsigned int i = 0; i < 6; i++)
            r = r / ysq + c[i];

        r = r / y;
        double corr = log(y);
        double spi = 0.9189385332046727417803297;
        return r + spi - 0.5*corr + y * (corr-1);
    }

    //should not happen
    assert(false);
    return 0.0;
}

double digamma(double x)
{
    const double large = 9.5;
    const double d1 = -0.5772156649015328606065121;  // digamma(1)
    const double d2 = (PI*PI)/6.0;
    const double small = 1e-6;
    const double s3 = 1.0/12.0;
    const double s4 = 1.0/120.0;
    const double s5 = 1.0/252.0;
    const double s6 = 1.0/240.0;
    const double s7 = 1.0/132.0;
//     const double s8 = 691.0/32760.0;
//     const double s9 = 1.0/12.0;
//     const double s10 = 3617.0/8160.0;

    double y = 0.0;

    if(x < 0.0)
        y = digamma(-x+1.0) + PI*cot(-PI*x);

    if (x == 0.0)
        y = -HUGE_VAL;

    if((x > 0.0) && (x <= small))
        y = y + d1 - 1.0/x + d2*x;

    // Reduce to digamma(X + N) where (X + N) >= large.
    while((x > small) && (x < large))
    {
        y = y - 1.0/x;
        x = x + 1.0;
    }

    // Use de Moivre's expansion if argument >= large.
    if(x >= large)
    {
        double r = 1.0/x;
        y = y + log(x) - 0.5*r;
        r = r*r;
        y = y - r*(s3 - r*(s4 - r*(s5 - r*(s6 - r*s7))));
    }

    return y;
}

vector<double> factlncache(101, -1.0);
double factln(unsigned int n)
{
    if (n < 2) return 0.0;

    if (n < 101)
    {
        if (factlncache[n] > 0.0)
            return factlncache[n];
        else
            return (factlncache[n] = gammaln(n + 1.0));
    }
    else
        return gammaln(n + 1.0);
}

double binom(unsigned int k, unsigned int n, double p)
{
    assert((p >= 0.0) && (p <= 1.0));

    if (n < k)
        return 0.0;
    else
    {
        double nk = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1);
        double lny = nk + k*log(p) + (n - k)*log(1 - p);
        return exp(lny);
    }
}

unsigned int solve_cubic(double o, double p, double q, double r, double &result0, double &result1, double &result2)
{
    double c2 = p / o;
    double c1 = q / o;
    double c0 = r / o;

    double a = (3.0 * c1 - c2 * c2) / 3.0;
    double b = (-9.0 * c1 * c2 + 27.0 * c0 + 2.0 * c2 * c2 * c2) / 27.0;
    double determinant = ((b * b) / 4.0) + ((a * a * a) / 27.0);

    double t[3] = {0.0, 0.0, 0.0};
    if (determinant > EPSILON)      //only one real root
    {
        double sqrtQ = sqrt(determinant);
        double temp1 = -b / 2.0 + sqrtQ;
        double temp2 = -b / 2.0 - sqrtQ;

        if (fabs(temp1) < EPSILON) temp1 = 0.0;
        if (fabs(temp2) < EPSILON) temp2 = 0.0;
        t[0] = cbrt(temp1) + cbrt(temp2);
    }
    else if (determinant < -EPSILON) //three distinct real roots
    {
        double theta = atan2(sqrt(-determinant), -b / 2.0);
        double phi = sqrt((b * b) / 4.0 - determinant);
        double cbrtPhi = cbrt(phi);
        double cosThetaOver3 = cos(theta / 3.0);
        double sinThetaOver3 = sin(theta / 3.0);
        double sqrt3 = sqrt(3.0);

        t[0] = 2.0 * cbrtPhi * cosThetaOver3;
        t[1] = -cbrtPhi * (cosThetaOver3 + sqrt3 * sinThetaOver3);
        t[2] = -cbrtPhi * (cosThetaOver3 - sqrt3 * sinThetaOver3);
    }
    else                        //three real roots, at least two are the same
    {
        t[0] = cbrt(b / 2.0);
        t[1] = t[0];
        t[2] = -2.0 * t[0];
    }

    t[0] -= c2 / 3.0;
    t[1] -= c2 / 3.0;
    t[2] -= c2 / 3.0;

    if (determinant > EPSILON)      //only one real root
    {
        result0 = t[0];
        result1 = 0.0;
        result2 = 0.0;
        return 1;
    }
    else if (determinant < -EPSILON) //three distinct real roots
    {
        while ((t[0] < t[1]) || (t[1] < t[2]))
        {
            if (t[0] < t[1]) swap(t[0], t[1]);
            if (t[1] < t[2]) swap(t[1], t[2]);
        }

        result0 = t[0];
        result1 = t[1];
        result2 = t[2];

        return 3;
    }
    else                        //three real roots, at least two are the same
    {
        if (t[0] > t[2])
        {
            result0 = t[0];
            result1 = t[0];
            result2 = t[2];
        }
        else if (t[0] < t[2])
        {
            result0 = t[2];
            result1 = t[0];
            result2 = t[0];
        }
        else
        {
            result0 = t[0];
            result1 = t[0];
            result2 = t[0];

            return 1;
        }

        return 2;
    }
}
