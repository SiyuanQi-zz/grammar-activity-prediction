#include "SignificantPattern.h"

#include <cassert>

using std::vector;
using std::string;
using std::ostringstream;

SignificantPattern::SignificantPattern()
{
}

SignificantPattern::SignificantPattern(const vector<unsigned int> &sequence)
{
    clear();
    for(unsigned int i = 0; i < sequence.size(); i++)
        push_back(sequence[i]);
}

SignificantPattern::~SignificantPattern()
{
}

unsigned int SignificantPattern::find(unsigned int unit) const
{
    for(unsigned int i = 0; i < size(); i++)
        if(at(i) == unit)
            return i;

    assert(false);
}

LexiconUnit* SignificantPattern::makeCopy() const
{
    return new SignificantPattern(*this);
}

string SignificantPattern::toString() const
{
    ostringstream sout;

    sout << "P[";
    if(size() > 0)
    {
        for(unsigned int i = 0; i < size() - 1; i++)
            sout << "P" << at(i) << " - ";
        if(size() > 0) sout << "P" << back();
    }
    sout << "]";

    return sout.str();
}
