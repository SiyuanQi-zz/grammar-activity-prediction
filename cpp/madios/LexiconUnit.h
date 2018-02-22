#ifndef LEXICONUNIT_H
#define LEXICONUNIT_H

#include "utils/Stringable.h"

class LexiconUnit: public Stringable
{
    public:
        virtual LexiconUnit* makeCopy() const = 0;
};

#endif
