#ifndef SIGNIFICANTPATTERN_H
#define SIGNIFICANTPATTERN_H

#include "LexiconUnit.h"

#include <vector>
#include <sstream>

class SignificantPattern: public LexiconUnit, public std::vector<unsigned int>
{
    public:
        SignificantPattern();
        explicit SignificantPattern(const std::vector<unsigned int> &sequence);
        virtual ~SignificantPattern();

        unsigned int find(unsigned int unit) const;

        virtual LexiconUnit* makeCopy() const;
        virtual std::string toString() const;
};

#endif
