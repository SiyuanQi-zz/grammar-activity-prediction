#ifndef JAMA_RQ_H
#define JAMA_RQ_H

#include "array1d.h"
#include "array2d.h"
#include "jama_qr.h"


using TNT::Array1D;
using TNT::Array2D;


namespace JAMA
{

template <class Real>
class RQ 
{
public:
    RQ(const Array2D<Real> &A)		/* constructor */
    : A_(A), QR_(transpose(flipud(A)))
	{
        R_ = fliplr(flipud(transpose(QR_.getR())));
        Q_ = flipud(transpose(QR_.getQ()));
    }
    
    Array2D<Real> getQ() const
    {
        return Q_;
    }
    
    Array2D<Real> getR() const
    {
        return R_;
    }

private:
   Array2D<Real> A_, Q_, R_;
   JAMA::QR<Real> QR_;
};

}

#endif

