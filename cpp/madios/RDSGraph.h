#ifndef RDSGRAPH_H
#define RDSGRAPH_H

#include "RDSNode.h"
#include "ADIOSUtils.h"
#include "maths/special.h"
#include "utils/MiscUtils.h"
#include "utils/TimeFuncs.h"
#include "ParseTree.h"

#include <algorithm>
#include <string>
#include <sstream>

// true, if both pvalues are less than alpha
bool isPatternSignificant(const SignificancePair &pvalues, double alpha);
bool operator<(const SignificancePair &a, const SignificancePair &b);

class RDSGraph: public Stringable
{
    public:
        RDSGraph();
        explicit RDSGraph(const std::vector<std::vector<std::string> > &sequences);

        std::vector<std::string> generate() const;
        std::vector<std::string> generate(const SearchPath &search_path) const;
        std::vector<std::string> generate(unsigned int node) const;
        void distill(const ADIOSParams &params);

        void convert2PCFG(std::ostream &out) const;
        void convert2nltkPCFG(std::ostream &out) const;

        virtual std::string toString() const;

    private:
        unsigned int corpusSize;
        std::vector<RDSNode> nodes;
        std::vector<SearchPath> paths;
        std::vector<ParseTree<unsigned int> > trees;

        // counts and normalised probabilities
        std::vector<std::vector<unsigned int> > counts;
        //std::vector<std::vector<double> > probs;

        void buildInitialGraph(const std::vector<std::vector<std::string> > &sequences);
        bool distill(const SearchPath &search_path, const ADIOSParams &params);
        bool generalise(const SearchPath &search_path, const ADIOSParams &params);

        // generalise and bootstrap
        EquivalenceClass computeEquivalenceClass(const SearchPath &search_path, unsigned int slotIndex);
        SearchPath bootstrap(std::vector<EquivalenceClass> &encountered_ecs, const SearchPath &search_path, double overlapThreshold) const;

        // compute matrix and pattern searching function
        void computeConnectionMatrix(ConnectionMatrix &connections, const SearchPath &search_path) const;
        void computeDescentsMatrix(Array2D<double> &flows, Array2D<double> &descents, const ConnectionMatrix &connections) const;
        bool findSignificantPatterns(std::vector<Range> &patterns, std::vector<SignificancePair> &pvalues, const ConnectionMatrix &connections, const Array2D<double> &flows, const Array2D<double> &descents, double eta, double alpha) const;

        // rewiring and update functions
        void updateAllConnections();
        void rewire(const std::vector<Connection> &connections, unsigned int ec);
        void rewire(const std::vector<Connection> &connections, const EquivalenceClass &ec);
        void rewire(const std::vector<Connection> &connections, const SignificantPattern &sp);
        std::vector<Connection> getRewirableConnections(const ConnectionMatrix &connections, const Range &bestSP, double alpha) const;

        // pattern searching auxiliary functions
        double computeRightSignificance(const ConnectionMatrix &connections, const Array2D<double> &flows, const std::pair<unsigned int, unsigned int> &descentPoint, double eta) const;
        double computeLeftSignificance(const ConnectionMatrix &connections, const Array2D<double> &flows, const std::pair<unsigned int, unsigned int> &descentPoint, double eta) const;
        double findBestRightDescentColumn(unsigned int &bestColumn, Array2D<double> &pvalueCache, const ConnectionMatrix &connections, const Array2D<double> &flows, const Array2D<double> &descents, const Range &pattern, double eta) const;
        double findBestLeftDescentColumn(unsigned int &bestColumn, Array2D<double> &pvalueCache, const ConnectionMatrix &connections, const Array2D<double> &flows, const Array2D<double> &descents, const Range &pattern, double eta) const;

        // auxilliary functions
        std::vector<Connection> filterConnections(const std::vector<Connection> &init_cons, unsigned int start_offset, const SearchPath &search_path) const;
        std::vector<Connection> getAllNodeConnections(unsigned int nodeIndex) const;
        unsigned int findExistingEquivalenceClass(const EquivalenceClass &ec);

        // counts the occurences of each lexicon unit
        void estimateProbabilities();

        // print functions
        std::string printSignificantPattern(const SignificantPattern &sp) const;
        std::string printEquivalenceClass(const EquivalenceClass &ec) const;
        std::string printNode(unsigned int node) const;
        std::string printPath(const SearchPath &path) const;
        std::string printNodeName(unsigned int node) const;
};

void printInfo(const ConnectionMatrix &connections, const Array2D<double> &flows, const Array2D<double> &descents);

#endif
