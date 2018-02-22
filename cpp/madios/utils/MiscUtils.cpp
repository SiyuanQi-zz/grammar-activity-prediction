#include "MiscUtils.h"

#include <sstream>
#include <algorithm>

using std::vector;
using std::string;
using std::istream;
using std::stringstream;


void getlines(istream &in, vector<string> &lines)
{
    while(!in.eof())
    {
        string line;
        getline(in, line);
        lines.push_back(line);
    }
}

vector<string> tokenise(const string &line)
{
    stringstream ss(line);

    vector<string> tokens;
    while(!ss.eof())
    {
        string tok;
        ss >> tok;

        if(tok.size() > 0)
            tokens.push_back(tok);
    }

    return tokens;
}

vector<string> tokenise(const string &line, const char &delimiter)
{
    stringstream ss(line);

    string tok;
    vector<string> tokens;
    while(getline(ss, tok, delimiter))
        if(tok.size() > 0)
            tokens.push_back(tok);

    return tokens;
}

string uppercase(const string &s)
{
    string result = s;
    transform(s.begin(), s.end(), result.begin(), toupper);
    return result;
}

string lowercase(const string &s)
{
    string result = s;
    transform(s.begin(), s.end(), result.begin(), tolower);
    return result;
}

string trimSpaces(const string &s)
{
    unsigned int t;
    string str = s;
    while ((t = str.find('\t')) != string::npos) str[t] = ' ';
    while ((t = str.find('\n')) != string::npos) str[t] = ' ';
    unsigned int n = str.find_first_not_of(" ");
    unsigned int k = str.find_last_not_of(" ");
    return str.substr(n, k - n + 1);
}
