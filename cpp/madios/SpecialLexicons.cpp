#include "SpecialLexicons.h"

using std::string;

LexiconUnit* StartSymbol::makeCopy() const
{
    return new StartSymbol();
}

string StartSymbol::toString() const
{
    return "START";
}

LexiconUnit* EndSymbol::makeCopy() const
{
    return new EndSymbol();
}

string EndSymbol::toString() const
{
    return "END";
}
