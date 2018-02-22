#ifndef SEARCHPATH_H
#define SEARCHPATH_H

#include "utils/Stringable.h"

#include <fstream>
#include <sstream>
#include <vector>

class SearchPath: public Stringable, public std::vector<unsigned int>
{
    public:
        SearchPath();
        explicit SearchPath(const std::vector<unsigned int> &path);
        virtual ~SearchPath();

        bool operator==(const SearchPath &other) const;
        void rewire(unsigned int start, unsigned int finish, unsigned int node);
        std::vector<unsigned int> operator()(unsigned int start, unsigned int finish) const;
        std::vector<unsigned int> substitute(unsigned int start, unsigned int finish, const std::vector<unsigned int> &segment) const;
        virtual std::string toString() const;
};

#endif
