#include "Stringable.h"

using std::ostream;

Stringable::Stringable()
{
}

Stringable::~Stringable()
{
}

ostream& operator<<(ostream &out, const Stringable &stringable)
{
    out << stringable.toString();
    return out;
}
