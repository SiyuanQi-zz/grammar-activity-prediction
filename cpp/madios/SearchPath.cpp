#include "SearchPath.h"

#include <cassert>

using std::string;
using std::vector;
using std::ostream;
using std::ostringstream;
using std::endl;

SearchPath::SearchPath()
{
}

SearchPath::SearchPath(const vector<unsigned int> &path)
:vector<unsigned int>(path)
{
}

SearchPath::~SearchPath()
{
}

bool SearchPath::operator==(const SearchPath &other) const
{
    if(size() != other.size())
        return false;

    for(unsigned int i = 0; i < size(); i++)
        if(at(i) != other[i])
            return false;

    return true;
}

void SearchPath::rewire(unsigned int start, unsigned int finish, unsigned int node)
{
    erase( begin()+start, begin()+finish+1);
    insert(begin()+start, node);
}

vector<unsigned int> SearchPath::operator()(unsigned int start, unsigned int finish) const
{
    assert(start <= finish);
    assert(finish< size());

    return vector<unsigned int>(begin() + start, begin() + finish + 1);
}

vector<unsigned int> SearchPath::substitute(unsigned int start, unsigned int finish, const vector<unsigned int> &segment) const
{
    assert(start <= finish);
    assert(finish < size());

    vector<unsigned int> new_path(begin(), begin()+start);
    new_path.insert(new_path.end(), segment.begin(), segment.end());
    new_path.insert(new_path.end(), begin()+finish+1, end());

    return new_path;
}

string SearchPath::toString() const
{
    ostringstream sout;
    sout << "[";
    for(unsigned int i = 0; i < size() - 1; i++)
        sout << at(i) << " --> ";
    sout << back() << "]";

    return sout.str();
}
