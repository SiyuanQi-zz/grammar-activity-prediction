#include "EquivalenceClass.h"

#include <algorithm>

using std::vector;
using std::string;
using std::ostringstream;

EquivalenceClass::EquivalenceClass()
{
}

EquivalenceClass::EquivalenceClass(const vector<unsigned int> &units)
:vector<unsigned int>(units)
{
}

EquivalenceClass::~EquivalenceClass()
{
}

EquivalenceClass EquivalenceClass::computeOverlapEC(const EquivalenceClass &other) const
{
    EquivalenceClass overlap;
    for(unsigned int i = 0; i < other.size(); i++)
        if(has(other[i]))
            overlap.add(other[i]);

    return overlap;
}

bool EquivalenceClass::has(unsigned int unit) const
{
    return (find(begin(), end(), unit) != end());
}

bool EquivalenceClass::add(unsigned int unit)
{
    if(has(unit))
        return false;

    push_back(unit);
    return true;
}

LexiconUnit* EquivalenceClass::makeCopy() const
{
    return new EquivalenceClass(*this);
}

string EquivalenceClass::toString() const
{
    ostringstream sout;

    sout << "E[";
    if(size() > 0)
    {
        for(unsigned int i = 0; i < size() - 1; i++)
            sout << "P" << at(i) << " | ";
        if(size() > 0) sout << "P" << back();
    }
    sout << "]";

    return sout.str();
}
