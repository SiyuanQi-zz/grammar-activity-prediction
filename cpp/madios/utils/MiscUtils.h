#ifndef MISC_UTILS_H
#define MISC_UTILS_H

#include <sstream>
#include <istream>
#include <string>
#include <vector>


void getlines(std::istream &in, std::vector<std::string> &lines);
std::vector<std::string> tokenise(const std::string &line);
std::vector<std::string> tokenise(const std::string &line, const char &delimiter);

std::string uppercase(const std::string &s);
std::string lowercase(const std::string &s);
std::string trimSpaces(const std::string &s);   //trim leading and trailing spaces

template <typename out_type, typename in_value>
out_type stream_cast(const in_value &t)
{
    std::stringstream ss;
    ss << t;
    out_type result;
    ss >> result;
    return result;
}

template <typename T>
std::string num2str(const T &num)
{
    return stream_cast<std::string, T>(num);
}

#endif
