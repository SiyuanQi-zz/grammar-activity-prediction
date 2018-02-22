#ifndef SPECIALLEXICONS_H
#define SPECIALLEXICONS_H

#include "LexiconUnit.h"

#include <string>

class StartSymbol: public LexiconUnit
{
    public:
        virtual LexiconUnit* makeCopy() const;
        virtual std::string toString() const;
};

class EndSymbol: public LexiconUnit
{
    public:
        virtual LexiconUnit* makeCopy() const;
        virtual std::string toString() const;
};

#endif
