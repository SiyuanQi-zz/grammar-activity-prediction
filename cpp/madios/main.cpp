#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include <fstream>

// Boost header files
#include <boost/filesystem.hpp>

#include "RDSGraph.h"
#include "utils/TimeFuncs.h"

namespace fs = boost::filesystem;
using std::vector;
using std::pair;
using std::string;
using std::stringstream;
using std::ifstream;
using std::ios;
using std::cout;
using std::endl;

vector<vector<string> > readSequencesFromFile(const string &filename)
{
    vector<vector<string> > sequences;
    vector<string> tokens;
    string token;

    ifstream in(filename.c_str(), ios::in);
    if(!in.is_open())
    {
        cout << "Unable to open file: " << filename << endl;
        exit(1);
    }

    while(!in.eof())
    {
        string line;
        getline(in, line);

        if(line.size() > 0)
        {
            stringstream ss(line);
            while(!ss.eof())
            {
                ss >> token;

                if(token == "*")
                    tokens.clear();
                else if(token == "#")
                {
                    sequences.push_back(tokens);
                    break;
                }
                else
                    tokens.push_back(token);
            }
        }
    }
    in.close();

    return sequences;
}

int main(int argc, char *argv[])
{
    if(argc < 6)
    {
        cout << "Usage:" << endl;
        cout << "ModifiedADIOS <filename> <eta> <alpha> <context_size> <coverage> ---OPTIONAL--- <number_of_new_sequences>" << endl;
        exit(1);
    }

    cout << "BEGIN CORPUS ----------" << endl;
    vector<vector<string> > sequences = readSequencesFromFile(argv[1]);
    for(unsigned int i = 0; i < sequences.size(); i++)
    {
        for(unsigned int j = 0; j < sequences[i].size(); j++)
            cout << sequences[i][j] << " ";
        cout << endl;
    }
    cout << "END CORPUS ----------" << endl << endl << endl;

    RDSGraph testGraph(sequences);
    cout << testGraph << endl;
    double startTime = getTime();
    testGraph.distill(ADIOSParams(atof(argv[2]), atof(argv[3]), atoi(argv[4]), atof(argv[5])));
    double endTime = getTime();
    cout << testGraph << endl << endl;

    std::cout << endl << "Time elapsed: " << endTime - startTime << " seconds" << endl << endl << endl << endl;

    fs::path corpusPath(argv[1]), grammarDir(corpusPath.parent_path()/".."/"grammar");
    fs::create_directories(grammarDir);
    fs::path grammarPath(grammarDir/fs::path(corpusPath.stem().string()+".pcfg"));
    std::ofstream fout(grammarPath.string());
    //testGraph.convert2PCFG(fout);
    testGraph.convert2nltkPCFG(fout);
/*
    startTime = getTime();
    testGraph.distill(ADIOSParams(atof(argv[2]), atof(argv[3])*10, atoi(argv[4])-2, atof(argv[5])));
    endTime = getTime();
    cout << testGraph << endl << endl;

    std::cout << endl << "Time elapsed: " << endTime - startTime << " seconds" << endl << endl << endl << endl;*/
/*
    vector<string> testString(sequences[10].begin(), sequences[10].end());
    for(unsigned int i = 0; i < testString.size() - 1; i++)
        std::cout << testString[i] << " ";
    std::cout << testString.back() << endl;
    SearchPath newPath = testGraph.encode(testString);
    std::cout << newPath << endl;
    testGraph.predict(newPath);

    if(argc > 6)
        for(unsigned int i = 0; i < static_cast<unsigned int>(atoi(argv[6])); i++)
        {
            vector<string> sequence = testGraph.generate();
            for(unsigned int j = 0; j < sequence.size(); j++)
                std::cout << sequence[j] << " ";
            std::cout << endl;
        }*/
}
