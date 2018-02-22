#ifndef ADIOSUTILS_H
#define ADIOSUTILS_H

#include "BasicSymbol.h"
#include "SpecialLexicons.h"
#include "SignificantPattern.h"
#include "EquivalenceClass.h"
#include "SearchPath.h"

class ADIOSParams
{
    public:
        double eta;
        double alpha;
        unsigned int contextSize;
        double overlapThreshold;

        ADIOSParams(double eta, double alpha, unsigned int contextSize, double overlapThreshold);
};

namespace LexiconTypes
{
enum LexiconEnum
{
    Start, End, Symbol, SP, EC
};
}

#endif
