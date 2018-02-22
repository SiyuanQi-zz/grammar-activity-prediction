#include "RDSGraph.h"

using std::min;
using std::max;
using std::vector;
using std::pair;
using std::string;
using std::ostream;
using std::ostringstream;
using std::endl;

bool isPatternSignificant(const SignificancePair &pvalues, double alpha)
{
    return (pvalues.first < alpha) && (pvalues.second < alpha);
}

bool operator<(const SignificancePair &a, const SignificancePair &b)
{
    double maxA = max(a.first, a.second);
    double maxB = max(b.first, b.second);

    return maxA < maxB;
}

ADIOSParams::ADIOSParams(double eta, double alpha, unsigned int contextSize, double overlapThreshold)
{
    assert((eta >= 0.0) && (eta <= 1.0));
    assert((alpha >= 0.0) && (alpha <= 1.0));

    this->eta = eta;
    this->alpha = alpha;
    this->contextSize = contextSize;
    this->overlapThreshold = overlapThreshold;
}

RDSGraph::RDSGraph()
{
    corpusSize = 0;
}

RDSGraph::RDSGraph(const vector<vector<string> > &sequences)
{
    srand(getSeedFromTime());
    buildInitialGraph(sequences);
}

void RDSGraph::distill(const ADIOSParams &params)
{
    std::cout << "eta = " << params.eta << endl;
    std::cout << "alpha = " << params.alpha << endl;
    std::cout << "contextSize = " << params.contextSize << endl;
    std::cout << "overlapThreshold = " << params.overlapThreshold << endl;

    while(true)
    {
        bool foundPattern = false;
        for(unsigned int i = 0; i < paths.size(); i++)
        {
            std::cout << "--------------------------- working on Path " << i << " of length " << paths[i].size() << " ----------------------------------" << endl;
            std::cout << printPath(paths[i]) << endl;

            if((params.contextSize < 3) || (paths[i].size() < params.contextSize))
            {
                bool foundAnotherPattern = distill(paths[i], params);
                foundPattern = foundAnotherPattern || foundPattern;
            }
            else
            {
                bool foundAnotherPattern = generalise(paths[i], params);
                foundPattern = foundAnotherPattern || foundPattern;
            }
        }
        if(!foundPattern)
            break;
    }

    estimateProbabilities();

    std::cout << endl << endl << endl;
    for(unsigned int i = 0; i < counts.size(); i++)
        if(counts[i].size() > 1)
        {
            std::cout << printNodeName(i);
            std::cout <<  " ---> [";
            for(unsigned int j = 0; j < counts[i].size(); j++)
            {
                std::cout << counts[i][j];
                if(j < (counts[i].size()-1))
                    std::cout << " | ";
            }
            std::cout << "]";/*
            std::cout <<  " ---> [";
            for(unsigned int j = 0; j < counts[i].size(); j++)
            {
                std::cout << counts[i][j];
                if(j < (counts[i].size()-1))
                    std::cout << " | ";
            }
            std::cout << "]";*/
            std::cout << endl;
        }
    std::cout << endl << endl << endl;
    trees[0].print(0, 0);
    std::cout << endl << endl << endl;
}

void RDSGraph::convert2PCFG(ostream &out) const
{
    out << "S _" << std::endl;

    for(unsigned int i = 0; i < nodes.size(); i++)
    {
        if(nodes[i].type == LexiconTypes::EC)
        {
            EquivalenceClass *ec = static_cast<EquivalenceClass *>(nodes[i].lexicon);
            for(unsigned int j = 0; j < ec->size(); j++)
                out << counts[i][j] << " E" << i << " --> " << printNodeName((*ec)[j]) << std::endl;
        }
        else if(nodes[i].type == LexiconTypes::SP)
        {
            SignificantPattern *sp = static_cast<SignificantPattern *>(nodes[i].lexicon);
            out << counts[i][0] << " P" << i << " -->";
            for(unsigned int j = 0; j < sp->size(); j++)
                out << " " << printNodeName((*sp)[j]);
            out << std::endl;
        }
    }

    for(unsigned int i = 0; i < paths.size(); i++)
    {
        out << "1 S -->";
        for(unsigned int j = 1; j < paths[i].size()-1; j++)
            out << " " << printNodeName(paths[i][j]);
        out << std::endl;
    }
}

void RDSGraph::convert2nltkPCFG(ostream &out) const
{
    /* print sentences first so nltk doesn't barf */
    for(unsigned int i = 0; i < paths.size(); i++)
    {
        out << "S ->";
        for(unsigned int j = 1; j < paths[i].size()-1; j++)
        {
            //out << " " << printNodeName(paths[i][j]);
            if (nodes[paths[i][j]].type == LexiconTypes::Symbol)
            {
                out << " \'" << printNodeName(paths[i][j]) << "\'";
            }else
            {
                out << " " << printNodeName(paths[i][j]);
            }

        }
        //out << " [" << 1.0/paths.size() << "]";
        out << std::endl;
    }

    for(unsigned int i = 0; i < nodes.size(); i++)
    {
        if(nodes[i].type == LexiconTypes::EC)
        {
            /* get total counts */
            EquivalenceClass *ec = static_cast<EquivalenceClass *>(nodes[i].lexicon);
            int total_count = 0;
            for(unsigned int j = 0; j < ec->size(); j++)
                total_count += counts[i][j];
            for(unsigned int j = 0; j < ec->size(); j++)
            {
                float probability = ((float) counts[i][j]) / total_count;
                //out << "E" << i << " -> " << printNodeName((*ec)[j]) << " [" << probability << "]" << std::endl;
                if (nodes[(*ec)[j]].type == LexiconTypes::Symbol)
                {
                    out << "E" << i << " -> \'" << printNodeName((*ec)[j]) << "\' [" << probability << "]" << std::endl;
                }else
                {
                    out << "E" << i << " -> " << printNodeName((*ec)[j]) << " [" << probability << "]" << std::endl;
                }
            }
        }
        else if(nodes[i].type == LexiconTypes::SP)
        {
            SignificantPattern *sp = static_cast<SignificantPattern *>(nodes[i].lexicon);
            out << "P" << i << " ->";
            for(unsigned int j = 0; j < sp->size(); j++)
            {
                //out << " " << printNodeName((*sp)[j]);
                if (nodes[(*sp)[j]].type == LexiconTypes::Symbol)
                {
                    out << " \'" << printNodeName((*sp)[j]) << "\'";
                }else
                {
                    out << " " << printNodeName((*sp)[j]);
                }
            }
            out << " [1.0]";
            out << std::endl;
        }
    }
}



vector<string> RDSGraph::generate() const
{
    unsigned int pathIndex = static_cast<unsigned int>(floor(uniform_rand() * paths.size()));
    return generate(paths[pathIndex]);
}

vector<string> RDSGraph::generate(const SearchPath &search_path) const
{
    vector<string> sequence;
    for(unsigned int i = 0; i < search_path.size(); i++)
    {
        vector<string> segment = generate(search_path[i]);
        sequence.insert(sequence.end(), segment.begin(), segment.end());
    }

    return sequence;
}

vector<string> RDSGraph::generate(unsigned int node) const
{
    vector<string> sequence;

    if(nodes[node].type == LexiconTypes::Start)
        sequence.push_back("*");
    else if(nodes[node].type == LexiconTypes::End)
        sequence.push_back("#");
    else if(nodes[node].type == LexiconTypes::Symbol)
        sequence.push_back((static_cast<BasicSymbol *>(nodes[node].lexicon))->getSymbol());
    else if(nodes[node].type == LexiconTypes::EC)
    {
        EquivalenceClass *ec = static_cast<EquivalenceClass *>(nodes[node].lexicon);
        unsigned int numberOfUnits = ec->size();
        unsigned int randomUnit = static_cast<unsigned int>(floor(numberOfUnits * uniform_rand()));
        vector<string> segment = generate(ec->at(randomUnit));
        sequence.insert(sequence.end(), segment.begin(), segment.end());
    }
    else if(nodes[node].type == LexiconTypes::SP)
    {
         SignificantPattern *SP = static_cast<SignificantPattern *>(nodes[node].lexicon);
         for(unsigned int i = 0; i < SP->size(); i++)
         {
             vector<string> segment = generate((*SP)[i]);
             sequence.insert(sequence.end(), segment.begin(), segment.end());
         }
    }
    else
        assert(false);

    assert(sequence.size() > 0);

    return sequence;
}

bool RDSGraph::distill(const SearchPath &search_path, const ADIOSParams &params)
{
    // look possible significant pattern found with help of equivalence class
    ConnectionMatrix connections;
    Array2D<double> flows, descents;
    computeConnectionMatrix(connections, search_path);
    computeDescentsMatrix(flows, descents, connections);

    vector<Range> patterns;
    vector<SignificancePair> pvalues;
    if(!findSignificantPatterns(patterns, pvalues, connections, flows, descents, params.eta, params.alpha))
        return false;

    SignificantPattern bestPattern(search_path(patterns.front().first, patterns.front().second));
    vector<Connection> connectionsToRewire = getRewirableConnections(connections, patterns.front(), params.alpha);
    rewire(connectionsToRewire, SignificantPattern(bestPattern));

    std::cout << "BEST PATTERN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    std::cout << "RANGE = [" << patterns.front().first << " " << patterns.front().second << "]" << endl;
    std::cout << bestPattern << " with " << "[" << pvalues.front().first << " " << pvalues.front().second << "]" << endl;
    std::cout << connectionsToRewire.size() << " connections rewired." << endl;
    std::cout << "END BEST PATTERN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;

    return true;
}

bool RDSGraph::generalise(const SearchPath &search_path, const ADIOSParams &params)
{
    // BOOTSTRAPPING STAGE
    // bootstrapping variables
    vector<Range> all_boosted_contexts;
    vector<SearchPath> all_boosted_paths;
    vector<vector<EquivalenceClass> > all_encountered_ecs;

    // initialise with just the search path with no bootstrapping
    all_boosted_contexts.push_back(Range(0, 0));
    all_boosted_paths.push_back(search_path);
    all_encountered_ecs.push_back(vector<EquivalenceClass>(max(static_cast<unsigned int>(0), params.contextSize-2)));

    // get all boosted paths
    for(unsigned int i = 0; (i+params.contextSize-1) < search_path.size(); i++)
    {
        Range context(i, i+params.contextSize-1);
        all_encountered_ecs.push_back(vector<EquivalenceClass>());
        SearchPath boosted_part = bootstrap(all_encountered_ecs.back(), SearchPath(search_path(context.first, context.second)), params.overlapThreshold);
        SearchPath boosted_path(search_path.substitute(context.first, context.second, boosted_part));
        all_boosted_contexts.push_back(context);
        all_boosted_paths.push_back(boosted_path);
    }



    // GENERALISATION STAGE
    // generalisation variables
    vector<unsigned int> general2boost;
    vector<unsigned int> all_general_slots;
    vector<SearchPath> all_general_paths;
    vector<EquivalenceClass> all_general_ecs;

    // initialise with just the search path with no generalisation
    general2boost.push_back(0);
    all_general_slots.push_back(0);
    all_general_paths.push_back(search_path);
    all_general_ecs.push_back(EquivalenceClass());

    // get all generalised paths
    for(unsigned int i = 1; i < all_boosted_paths.size(); i++)
    {
        unsigned int context_start = all_boosted_contexts[i].first;
        unsigned int context_finish = all_boosted_contexts[i].second;
        SearchPath boosted_part(all_boosted_paths[i](context_start, context_finish));

        // try all the possible slots
        unsigned int start_index = all_general_paths.size();
        for(unsigned int j = 1; j < params.contextSize-1; j++)
        {
            EquivalenceClass ec = computeEquivalenceClass(boosted_part, j);

            // test that the found equivalence class actually has more than one element
            SearchPath general_path = all_boosted_paths[i];
            if(ec.size() > 1)   // check if found equivalence class is very similar to an existing EC, need to double check
                general_path[context_start+j] = findExistingEquivalenceClass(ec);

            // if general_path is the same as the original search path, no need to test it
            if(general_path == search_path)
                continue;

            // if general_path is already one of the path found for this boosted path, no need to test it
            bool repeated = false;
            for(unsigned int k = start_index; k < all_general_paths.size(); k++)
                if(general_path == all_general_paths[k])
                {
                    repeated = true;
                    break;
                }
            if(repeated) continue;

            // added the generalised path sto the list to be tested
            general2boost.push_back(i);  // add the boosted path number corresponding to the general path
            all_general_slots.push_back(context_start+j);  // stores the slot that was generalised
            all_general_paths.push_back(general_path);
            all_general_ecs.push_back(ec);
        }
    }
    std::cout << all_general_paths.size() << " paths tested" << endl;



    // DISTILLATION STAGE
    // significant pattern variables
    vector<Range> all_patterns;
    vector<SignificancePair> all_pvalues;
    vector<unsigned int> pattern2general;

    // test each path for significant patterns;
    RDSGraph temp_graph(*this);
    for(unsigned int i = 0; i < all_general_paths.size(); i++)
    {
        ConnectionMatrix connections;
        unsigned int slot_index = all_general_slots[i];
        if(all_general_paths[i][slot_index] >= nodes.size()) // if a new EC is expected, temporarily rewire the RDSGraph
        {
            temp_graph.rewire(vector<Connection>(), EquivalenceClass(all_general_ecs[i]));
            temp_graph.computeConnectionMatrix(connections, all_general_paths[i]);
            temp_graph.nodes.pop_back();
            temp_graph.updateAllConnections();
        }
        else
            computeConnectionMatrix(connections, all_general_paths[i]);

        // compute flows and descents matrix from connection matrix
        Array2D<double> flows, descents;
        computeDescentsMatrix(flows, descents, connections);

        // look for significant patterns
        vector<Range> some_patterns;
        vector<SignificancePair> some_pvalues;
        if(!findSignificantPatterns(some_patterns, some_pvalues, connections, flows, descents, params.eta, params.alpha))
            continue;

        // add them to the list
        for(unsigned int j = 0; j < some_patterns.size(); j++)
//         for(unsigned int j = 0; j < 1; j++) // just take the best pattern at the moment, use all candidate patterns later
        {   // only accept the pattern if the any completely new equivalence class is in the distilled pattern
            if(all_general_paths[i][all_general_slots[i]] >= nodes.size())
                if((all_general_slots[i] < some_patterns[j].first) || (all_general_slots[i] > some_patterns[j].second))
                    continue;

            all_patterns.push_back(some_patterns[j]);
            all_pvalues.push_back(some_pvalues[j]);
            pattern2general.push_back(i);
        }
    }



    // LOOK FOR MOST SIGNIFICANT PATTERNS
    bool best_pattern_found = false;
    unsigned int best_pattern_index = all_patterns.size();
    for(unsigned int i = 0; i < all_patterns.size(); i++)
    {
        unsigned int current_pattern_length = all_patterns[i].second - all_patterns[i].first;
        if(best_pattern_found)
        {
            unsigned int best_pattern_length = all_patterns[best_pattern_index].second - all_patterns[best_pattern_index].first;
//             if(current_pattern_length >= best_pattern_length)
//             {
//                 if(current_pattern_length == best_pattern_length)
//                 {
                    if(!(all_pvalues[i] < all_pvalues[best_pattern_index]))
                        continue;
//                 }
//                 else
//                     continue;
//             }
        }

        best_pattern_found = true;
        best_pattern_index = i;
    }
    if(!best_pattern_found)
        return false;
    assert(best_pattern_index < all_patterns.size());
    std::cout << all_patterns.size() << " patterns found" << endl;

    // get alll the information about the best pattern
    Range best_pattern = all_patterns[best_pattern_index];
    SignificancePair best_pvalues = all_pvalues[best_pattern_index];

    unsigned int best_general_index = pattern2general[best_pattern_index];
    SearchPath best_path = all_general_paths[best_general_index];
    //unsigned int best_slot = all_general_slots[best_general_index];
    EquivalenceClass best_ec = all_general_ecs[best_general_index];

    unsigned int best_boosted_index = general2boost[best_general_index];
    Range best_context = all_boosted_contexts[best_boosted_index];
    vector<EquivalenceClass> best_encountered_ecs = all_encountered_ecs[best_boosted_index];



    // REWIRING STAGE
    std::cout << "STARTS REWIRING" << endl;
    unsigned int old_num_nodes = nodes.size();
    unsigned int search_start = max(best_pattern.first, best_context.first);
    unsigned int search_finish = min(best_pattern.second, best_context.second);
    for(unsigned int i = search_start; i <= search_finish; i++)
    {
        if(best_path[i] >= old_num_nodes)       // true if a new EC was discovered at the specific slot
        {
            best_path[i] = nodes.size();
            rewire(vector<Connection>(), EquivalenceClass(best_ec));
        }
        else if(best_path[i] != search_path[i]) // true if the part of the context was boosted from existing ECs
        {
            unsigned int local_slot = i - (best_context.first + 1);
            EquivalenceClass *best_exisiting_ec = static_cast<EquivalenceClass *>(nodes[best_path[i]].lexicon);
            EquivalenceClass overlap_ec = best_encountered_ecs[local_slot].computeOverlapEC(*best_exisiting_ec);
            double overlap_ratio = overlap_ec.size() / best_exisiting_ec->size();

            if(overlap_ratio < 1.0)            // true if the overlap with existing EC is less than 1.0, only use the subset that overlaps with it
            {
                std::cout << "NEW OVERLAP EC USED: E[" << printEquivalenceClass(overlap_ec) << "]" << endl;
                best_path[i] = nodes.size();
                rewire(vector<Connection>(), EquivalenceClass(overlap_ec));
            }
            else
            {
                std::cout << "OLD OVERLAP EC USED: E[" << printNode(best_path[i]) << "]" << endl;
                //rewire(vector<Connection>(), best_path[i]);
            }
        }
    }
    ConnectionMatrix best_connections;
    computeConnectionMatrix(best_connections, best_path);
    vector<Connection> best_pattern_connections = getRewirableConnections(best_connections, best_pattern, params.alpha);
    rewire(best_pattern_connections, SignificantPattern(best_path(best_pattern.first, best_pattern.second)));
    std::cout << best_pattern_connections .size() << " occurences rewired" << endl;
    std::cout << "ENDS REWIRING" << endl;

    return true;
}

string RDSGraph::toString() const
{
    ostringstream sout;

    sout << "Search Paths" << endl;
    for(unsigned int i = 0; i < paths.size(); i++)
        sout << printPath(paths[i]) << endl;

    sout << endl << "RDS Graph Nodes " << nodes.size() << endl;
    for(unsigned int i = 0; i < nodes.size(); i++)
    {
        sout << "Lexicon " << i << ": " << printNode(i) << "   ------->  " << nodes[i].parents.size() << "  [";
        for(unsigned int j = 0; j < nodes[i].parents.size(); j++)
        {
            sout << nodes[i].parents[j].first;// << "." << nodes[i].parents[j].second;
            if(j < (nodes[i].parents.size() - 1)) sout << "   ";
        }
        sout << "]" << endl;
    }

    return sout.str();
}

void RDSGraph::buildInitialGraph(const vector<vector<string> > &sequences)
{   //pad the temporary lexicon vector with empty element to acount for special start and end state
    vector<string> lexicon;
    lexicon.push_back("");
    lexicon.push_back("");

    //insert the special symbols
    nodes.push_back(RDSNode(new StartSymbol(), LexiconTypes::Start));
    nodes.push_back(RDSNode(new EndSymbol(), LexiconTypes::End));
    for(unsigned int i = 0; i < sequences.size(); i++)
    {
        vector<unsigned int> currentPath;

        //insert start state
        currentPath.push_back(0);

        //create the main part of the graph
        for(unsigned int j = 0; j < sequences[i].size(); j++)
        {
            vector<string>::iterator foundPosition = find(lexicon.begin(), lexicon.end(), sequences[i][j]);
            if(foundPosition == lexicon.end())
            {
                lexicon.push_back(sequences[i][j]);
                nodes.push_back(RDSNode(new BasicSymbol(sequences[i][j]), LexiconTypes::Symbol));
                currentPath.push_back(lexicon.size() - 1);
            }
            else
                currentPath.push_back(foundPosition - lexicon.begin()); //+2 offset for start and end states
        }

        //insert end state
        currentPath.push_back(1);

        paths.push_back(SearchPath(currentPath));
    }

    updateAllConnections();

    // create initial parse trees
    for(unsigned int i = 0; i < paths.size(); i++)
        trees.push_back(ParseTree<unsigned int>(paths[i]));
}

void RDSGraph::computeConnectionMatrix(ConnectionMatrix &connections, const SearchPath &search_path) const
{
    // calculate subpath distributions, symmetrical matrix
    unsigned dim = search_path.size();
    connections = ConnectionMatrix(dim, dim);
    for(unsigned int i = 0; i < dim; i++)
    {
        connections(i, i) = getAllNodeConnections(search_path[i]);

        // compute the column from the diagonal
        for(unsigned int j = i + 1; j < dim; j++)
        {
            connections(j, i) = filterConnections(connections(j - 1, i), j-i, SearchPath(search_path(j, j)));
            connections(i, j) = connections(j, i);
        }
    }
}

EquivalenceClass RDSGraph::computeEquivalenceClass(const SearchPath &search_path, unsigned int slotIndex)
{
    assert(0 < slotIndex);
    assert(slotIndex < (search_path.size()-1));

    // get the candidate connections
    vector<Connection> equivalenceConnections = getAllNodeConnections(search_path[0]);
    equivalenceConnections = filterConnections(equivalenceConnections, 0,           SearchPath(search_path(0, slotIndex-1)));
    equivalenceConnections = filterConnections(equivalenceConnections, slotIndex+1, SearchPath(search_path(slotIndex+1, search_path.size()-1)));

    //build equivalence class
    EquivalenceClass ec;
    for(unsigned int i = 0; i < equivalenceConnections.size(); i++)
    {
        unsigned int currentPath = equivalenceConnections[i].first;
        unsigned int currentStart = equivalenceConnections[i].second;

        equivalenceConnections[i].second = currentStart + slotIndex;
        ec.add(paths[currentPath][equivalenceConnections[i].second]);
    }

    return ec;
}

SearchPath RDSGraph::bootstrap(vector<EquivalenceClass> &encountered_ecs, const SearchPath &search_path, double overlapThreshold) const
{
    // find all possible connections
    vector<Connection> equivalenceConnections = filterConnections(getAllNodeConnections(search_path[0]), search_path.size()-1, SearchPath(search_path(search_path.size()-1, search_path.size()-1)));

    // find potential ECs
    encountered_ecs.clear();
    for(unsigned int i = 1; i < search_path.size()-1; i++)
    {
        encountered_ecs.push_back(EquivalenceClass());
        for(unsigned int j = 0; j < equivalenceConnections.size(); j++)
        {
            unsigned int currentPath = equivalenceConnections[j].first;
            unsigned int currentStart = equivalenceConnections[j].second;

            encountered_ecs.back().add(paths[currentPath][currentStart+i]);
        }
    }

    // init bootstrap data
    vector<unsigned int> overlap_ecs = search_path(1, search_path.size()-2);
    vector<double> overlap_ratios(search_path.size()-2, 0.0);

    // bootstrap search path
    SearchPath bootstrap_path = search_path;
    for(unsigned int i = 0; i < encountered_ecs.size(); i++)
    {
        for(unsigned int j = 0; j < nodes.size(); j++)
            if(nodes[j].type == LexiconTypes::EC)
            {
                EquivalenceClass *ec = static_cast<EquivalenceClass *>(nodes[j].lexicon);
                double overlap = encountered_ecs[i].computeOverlapEC(*ec).size()/static_cast<double>(ec->size());
                if((overlap > overlap_ratios[i]) && (overlap > overlapThreshold))
                {
                    overlap_ecs[i] = j;
                    overlap_ratios[i] = overlap;
                }
            }
        bootstrap_path[i + 1] = overlap_ecs[i];
    }

    return bootstrap_path;
}

void RDSGraph::computeDescentsMatrix(Array2D<double> &flows, Array2D<double> &descents, const ConnectionMatrix &connections) const
{
    // calculate P_R and P_L
    unsigned dim = connections.dim1();
    flows = Array2D<double>(dim, dim, -1.0);
    for(unsigned int i = 0; i < dim; i++)
        for(unsigned int j = 0; j < dim; j++)
            if(i > j)
                flows(i, j) = static_cast<double>(connections(i, j).size()) / connections(i-1, j).size();
            else if(i < j)
                flows(i, j) = static_cast<double>(connections(i, j).size()) / connections(i+1, j).size();
            else
                flows(i, j) = static_cast<double>(connections(i, j).size()) / corpusSize;

    // calculate D_R and D_L
    descents = Array2D<double>(dim, dim, -1.0);
    for(unsigned int i = 0; i < dim; i++)
        for(unsigned int j = 0; j < dim; j++)
            if(i > j)
                descents(i, j) = flows(i, j) / flows(i-1, j);
            else if(i < j)
                descents(i, j) = flows(i, j) / flows(i+1, j);
            else
                descents(i, j) = 1.0;
}

bool RDSGraph::findSignificantPatterns(vector<Range> &patterns, vector<SignificancePair> &pvalues, const ConnectionMatrix &connections, const Array2D<double> &flows, const Array2D<double> &descents, double eta, double alpha) const
{
    patterns.clear();
    pvalues.clear();

    //find candidate pattern start and ends
    unsigned int pathLength = descents.dim1();
    vector<unsigned int> candidateEndRows;
    vector<unsigned int> candidateStartRows;
    for(int i = 0; i < descents.dim1(); i++)
    {
        for(int j = i - 1; j >= 0; j--)
            if(descents(i, j) < eta)
            {
                candidateEndRows.push_back(i - 1);
                break;
            }

        for(int j = i + 1; j < descents.dim2(); j++)
            if(descents(i, j) < eta)
            {
                candidateStartRows.push_back(i + 1);
                break;
            }
    }

    //find candidate patterns;
    vector<Range> candidatePatterns;
    for(unsigned int i = 0; i < candidateStartRows.size(); i++)
        for(unsigned int j = 0; j < candidateEndRows.size(); j++)
            if(candidateStartRows[i] < candidateEndRows[j])
                candidatePatterns.push_back(Range(candidateStartRows[i], candidateEndRows[j]));

    //for(unsigned int i = 0; i < candidatePatterns.size(); i++)
    //    std::cout << "Candidate Pattern " << i << " = " << candidatePatterns[i].first << " " << candidatePatterns[i].second << endl;

    Array2D<double> pvalueCache(pathLength, pathLength, 2.0);
    for(unsigned int i = 0; i < candidatePatterns.size(); i++)
    {   //std::cout << "START+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        //std::cout << "Testing pattern at [" << candidatePatterns[i].first << " -> " << candidatePatterns[i].second << "]" << endl;

        SignificancePair tempPvalues(2.0, 2.0);
        pair<unsigned int, unsigned int> descentContexts;
        tempPvalues.second = findBestRightDescentColumn(descentContexts.second, pvalueCache, connections, flows, descents, candidatePatterns[i], eta);
        tempPvalues.first = findBestLeftDescentColumn(descentContexts.first, pvalueCache, connections, flows, descents, candidatePatterns[i], eta);
        if((fabs(tempPvalues.first) > 1.0) || (fabs(tempPvalues.second) > 1.0)) continue;

        //std::cout << "P_R at " << candidatePatterns[i].second << " " << descentContexts.second << " = " << flows(candidatePatterns[i].second, descentContexts.second) << endl;
        //std::cout << "right pvalue = " << 1.0 - tempPvalues.second << " for " << connections(candidatePatterns[i].second + 1, descentContexts.second).size() << " out of " << connections(candidatePatterns[i].second, descentContexts.second).size() << endl;

        //std::cout << "P_L at " << candidatePatterns[i].first << " " << descentContexts.first << " = " << flows(candidatePatterns[i].first, descentContexts.first) << endl;
        //std::cout << "left pvalue = " << 1.0 - tempPvalues.first << " for " << connections(candidatePatterns[i].first - 1, descentContexts.first).size() << " out of " << connections(candidatePatterns[i].first, descentContexts.first).size() << endl;

        // pattern IS significant
        if(isPatternSignificant(tempPvalues, alpha))
        {
            //std::cout << "Pattern is significant at [" << 1-tempPvalues.first << " --- " << 1-tempPvalues.second << "]" << endl;
            //std::cout << "Right descent context is [" << descentContexts.second << " -> " << candidatePatterns[i].second << "]" << endl;
            //std::cout << "Left descent context is [" << descentContexts.first << " -> " << candidatePatterns[i].first << "]" << endl;

            patterns.push_back(candidatePatterns[i]);
            pvalues.push_back(tempPvalues);

            //found a MORE significant pattern
            if((patterns.size() == 1) || (pvalues.back() < pvalues.front()))
            {
                swap(patterns.front(), patterns.back());
                swap(pvalues.front(), pvalues.back());
                //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!found a better SP" << endl;
            }
        }
        //std::cout << "END----------------------------------------------------------------------------------------------" << endl;
    }

    return patterns.size() > 0;
}

vector<Connection> RDSGraph::getRewirableConnections(const ConnectionMatrix &connections, const Range &bestSP, double alpha) const
{
    vector<Connection> validConnections = connections(bestSP.second, bestSP.first);

    return validConnections;
}

void RDSGraph::rewire(const std::vector<Connection> &connections, unsigned int ec)
{
    assert(nodes[ec].type == LexiconTypes::EC);

    for(unsigned int i = 0; i < connections.size(); i++)
        paths[connections[i].first][connections[i].second] = ec;

    updateAllConnections();
}

void RDSGraph::rewire(const vector<Connection> &connections, const EquivalenceClass &ec)
{
    nodes.push_back(RDSNode(new EquivalenceClass(ec), LexiconTypes::EC));
    rewire(connections, nodes.size() - 1);
}

void RDSGraph::rewire(const vector<Connection> &connections, const SignificantPattern &sp)
{
    nodes.push_back(RDSNode(new SignificantPattern(sp), LexiconTypes::SP));
    const SignificantPattern &pattern = sp;

    assert(connections.size() > 0);
    unsigned int pattern_size = pattern.size();

    // remove any overlapping connections
    vector<Connection> sorted_connections;
    for(unsigned int i = 0; i < connections.size(); i++)
    {
        unsigned int current_path_index = connections[i].first;
        unsigned int current_path_pos = connections[i].second;

        bool found_group = false, inserted = false;
        for(unsigned int j = 0; j < sorted_connections.size(); j++)
            if(current_path_index == sorted_connections[j].first)
            {
                found_group = true;
                if(current_path_pos < sorted_connections[j].second)
                {
                    sorted_connections.insert(sorted_connections.begin()+j, connections[i]);
                    inserted = true;
                    break;
                }
            }
            else if(found_group)
            {
                sorted_connections.insert(sorted_connections.begin()+j, connections[i]);
                inserted = true;
                break;
            }

        if(!inserted)
            sorted_connections.push_back(connections[i]);
    }

    // validate the sorted connections
    vector<Connection> valid_connections;
    valid_connections.push_back(sorted_connections.front());
    for(unsigned int i = 1; i < sorted_connections.size(); i++)
    {
        unsigned int current_path_index = sorted_connections[i].first;
        unsigned int current_path_pos = sorted_connections[i].second;
        unsigned int last_path_index = valid_connections.back().first;
        unsigned int last_path_pos = valid_connections.back().second;

        // the path is the same as the last path the pattern overlaps with the last pattern then do not rewire it
        if((current_path_index == last_path_index) && (current_path_pos <= (last_path_pos+pattern_size-1)))
            continue;

        valid_connections.push_back(sorted_connections[i]);
    }
    std::cout << valid_connections.size() << " valid_connections" << endl;

    // rewire the connections in reverse order to avoid problems with path changing size
    for(unsigned int i = valid_connections.size()-1; i < valid_connections.size(); i--)
    {
        unsigned int path_index = valid_connections[i].first;
        unsigned int path_pos = valid_connections[i].second;

        // rewiring the parse trees
        SearchPath segment(paths[path_index](path_pos, path_pos+pattern_size-1));
        for(unsigned int j = 0; j < segment.size(); j++)
            if(segment[j] != pattern[j])
                trees[path_index].rewire(path_pos+j, path_pos+j, pattern[j]);
        trees[path_index].rewire(path_pos, path_pos+pattern_size-1, nodes.size()-1);

        // rewiring the paths
        paths[path_index].rewire(path_pos, path_pos+pattern_size-1, nodes.size()-1);
    }

    updateAllConnections();
}

void RDSGraph::updateAllConnections()
{
    for(unsigned int i = 0; i < nodes.size(); i++)
    {
        nodes[i].setConnections(vector<Connection>());
        nodes[i].parents.clear();
    }

    corpusSize = 0;
    for(unsigned int i = 0; i < paths.size(); i++)
    {
        corpusSize += paths[i].size();
         for(unsigned int j = 0; j < paths[i].size(); j++)
             nodes[paths[i][j]].addConnection(Connection(i, j));
    }

    for(unsigned int i = 0; i < nodes.size(); i++)
    {
        if(nodes[i].type == LexiconTypes::SP)
        {
            SignificantPattern *sp = static_cast<SignificantPattern *>(nodes[i].lexicon);
            for(unsigned int j = 0; j < sp->size(); j++)
                nodes[sp->at(j)].parents.push_back(Connection(i, sp->find(sp->at(j))));
        }
        else if(nodes[i].type == LexiconTypes::EC)
        {
            EquivalenceClass *ec = static_cast<EquivalenceClass *>(nodes[i].lexicon);
            for(unsigned int j = 0; j < ec->size(); j++)
                nodes[ec->at(j)].parents.push_back(Connection(i, 0));
        }
    }
}

double RDSGraph::computeRightSignificance(const ConnectionMatrix &connections, const Array2D<double> &flows, const pair<unsigned int, unsigned int> &descentPoint, double eta) const
{
    unsigned int row = descentPoint.first;
    unsigned int col = descentPoint.second;
    assert(row > col);

    double significance = 0.0;
    unsigned int patternOccurences = connections(row - 1, col).size();
    unsigned int descentOccurences = connections(row, col).size();
    for(unsigned int i = 0; i <= descentOccurences; i++)
        significance += binom(i, patternOccurences, eta * flows(row - 1, col));

    return min(max(significance, 0.0), 1.0);
}

double RDSGraph::computeLeftSignificance(const ConnectionMatrix &connections, const Array2D<double> &flows, const pair<unsigned int, unsigned int> &descentPoint, double eta) const
{
    unsigned int row = descentPoint.first;
    unsigned int col = descentPoint.second;
    assert(row < col);

    double significance = 0.0;
    unsigned int patternOccurences = connections(row + 1, col).size();
    unsigned int descentOccurences = connections(row, col).size();
    for(unsigned int i = 0; i <= descentOccurences; i++)
        significance += binom(i, patternOccurences, eta * flows(row + 1, col));

    return min(max(significance, 0.0), 1.0);;
}

double RDSGraph::findBestRightDescentColumn(unsigned int &bestColumn, Array2D<double> &pvalueCache, const ConnectionMatrix &connections, const Array2D<double> &flows, const Array2D<double> &descents, const Range &pattern, double eta) const
{
    double pvalue = 2.0;
    pair<unsigned int, unsigned int> descentPoint(pattern.second + 1, bestColumn);
    for(unsigned int i = 0; i <= pattern.first; i++)
    {
        descentPoint.second = i;
        if(!(descents(descentPoint.first, descentPoint.second) < eta)) continue;
        if(pvalueCache(pattern.second + 1, i) > 1.0)
            pvalueCache(pattern.second + 1, i) = computeRightSignificance(connections, flows, descentPoint, eta);

        if(pvalueCache(pattern.second + 1, i) < pvalue)
        {
            bestColumn = i;
            pvalue = pvalueCache(pattern.second + 1, i);
        }
    }

    return pvalue;
}

double RDSGraph::findBestLeftDescentColumn(unsigned int &bestColumn, Array2D<double> &pvalueCache, const ConnectionMatrix &connections, const Array2D<double> &flows, const Array2D<double> &descents, const Range &pattern, double eta) const
{
    double pvalue = 2.0;
    pair<unsigned int, unsigned int> descentPoint(pattern.first - 1, bestColumn);
    for(int i = pattern.second; i < connections.dim2(); i++)
    {
        descentPoint.second = i;
        if(!(descents(descentPoint.first, descentPoint.second) < eta)) continue;
        if(pvalueCache(pattern.first - 1, i) > 1.0)
            pvalueCache(pattern.first - 1, i) = computeLeftSignificance(connections, flows, descentPoint, eta);

        if(pvalueCache(pattern.first - 1, i) < pvalue)
        {
            bestColumn = i;
            pvalue = pvalueCache(pattern.first - 1, i);
        }
    }

    return pvalue;
}

vector<Connection> RDSGraph::filterConnections(const vector<Connection> &init_cons, unsigned int start_offset, const SearchPath &search_path) const
{
    vector<Connection> filtered_cons;
    for(unsigned int i = 0; i < init_cons.size(); i++)
    {
        unsigned int cur_path = init_cons[i].first;
        unsigned int cur_pos = init_cons[i].second;

        // discard current connection because the path is not long enough to match the search path (segment)
        if((cur_pos+start_offset+search_path.size()) > paths[cur_path].size())
            continue;

        unsigned int count = search_path.size();
        for(unsigned int j = 0; j < search_path.size(); j++)
        {
            unsigned int actual_pos = j+cur_pos+start_offset;
            if(nodes[search_path[j]].type == LexiconTypes::EC)
            {   // if node on search path is EC and it contains the node and temp path
                if(!(static_cast<EquivalenceClass *>(nodes[search_path[j]].lexicon)->has(paths[cur_path][actual_pos])))
                    break;
            }
            else// else just test if they are the same node (BasicSymbol)
                if(search_path[j] != paths[cur_path][actual_pos])
                    break;

            count--;
        }

        // 0 if search_path completely matches temp_path
        if(count == 0)
            filtered_cons.push_back(init_cons[i]);
    }

    return filtered_cons;
}

vector<Connection> RDSGraph::getAllNodeConnections(unsigned int nodeIndex) const
{
    vector<Connection> connections = nodes[nodeIndex].getConnections();

    //get all connections belonging to the nodes in the equivalence class
    if(nodes[nodeIndex].type == LexiconTypes::EC)
    {
        EquivalenceClass *ec = static_cast<EquivalenceClass *>(nodes[nodeIndex].lexicon);
        for(unsigned int i = 0; i < ec->size(); i++)
        {
            vector<Connection> tempConnections = nodes[ec->at(i)].getConnections();
            connections.insert(connections.end(), tempConnections.begin(), tempConnections.end());
        }
    }

    return connections;
}

unsigned int RDSGraph::findExistingEquivalenceClass(const EquivalenceClass &ec)
{   // look for the existing ec that is a subset of the given ec
    for(unsigned int i = 0; i < nodes.size(); i++)
        if(nodes[i].type == LexiconTypes::EC)
        {
            EquivalenceClass *temp_ec = static_cast<EquivalenceClass *>(nodes[i].lexicon);
            if(ec.computeOverlapEC(*temp_ec).size() == temp_ec->size())
                return i;
        }

    return nodes.size();
}

void RDSGraph::estimateProbabilities()
{
    for(unsigned int i = 0; i < nodes.size(); i++)
        if(nodes[i].type == LexiconTypes::EC)
        {
            EquivalenceClass *ec = static_cast<EquivalenceClass *>(nodes[i].lexicon);
            counts.push_back(vector<unsigned int>(ec->size(), 0));
        }
        else
            counts.push_back(vector<unsigned int>(1, 0));

    for(unsigned int i = 0; i < trees.size(); i++)
    {
        const vector<ParseNode<unsigned int> > &tree_nodes = trees[i].nodes();
        for(unsigned int j = 1; j < tree_nodes.size(); j++)
        {
            unsigned int node_index = tree_nodes[j].value();
            if(nodes[node_index].type == LexiconTypes::EC)
            {
                assert(tree_nodes[j].children().size() == 1);

                EquivalenceClass *ec = static_cast<EquivalenceClass *>(nodes[node_index].lexicon);
                unsigned int first_child_pos = tree_nodes[j].children().front();
                unsigned int first_child_val = tree_nodes[first_child_pos].value();

                for(unsigned int k = 0; k < ec->size(); k++)
                    if(ec->at(k) == first_child_val)
                        counts[node_index][k]++;
            }
            else
                counts[node_index][0]++;
        }
    }
}

string RDSGraph::printSignificantPattern(const SignificantPattern &sp) const
{
    ostringstream sout;

    for(unsigned int i = 0; i < sp.size(); i++)
    {
        unsigned tempIndex = sp[i];
        //assert(tempIndex < nodes.size());
        if(nodes[tempIndex].type == LexiconTypes::EC)
            sout << "E" << tempIndex;
        else if(nodes[tempIndex].type == LexiconTypes::SP)
            sout << "P" << tempIndex;
        else
            sout << *(nodes[tempIndex].lexicon);

        if(i < (sp.size() - 1)) sout << " - ";
    }

    return sout.str();
}

string RDSGraph::printEquivalenceClass(const EquivalenceClass &ec) const
{
    ostringstream sout;

    for(unsigned int i = 0; i < ec.size(); i++)
    {
        unsigned tempIndex = ec[i];
        if(nodes[tempIndex].type == LexiconTypes::EC)
            sout << "E" << tempIndex;
        else if(nodes[tempIndex].type == LexiconTypes::SP)
            sout << "P" << tempIndex;
        else
            sout << *(nodes[tempIndex].lexicon);

        if(i < (ec.size() - 1)) sout << " | ";
    }

    return sout.str();
}

string RDSGraph::printNode(unsigned int node) const
{
    ostringstream sout;

    if(nodes[node].type == LexiconTypes::EC)
    {
        EquivalenceClass *ec = static_cast<EquivalenceClass *>(nodes[node].lexicon);
        sout << "E[" << printEquivalenceClass(*ec) << "]";
    }
    else if(nodes[node].type == LexiconTypes::SP)
    {
        SignificantPattern *sp = static_cast<SignificantPattern *>(nodes[node].lexicon);
        sout << "P[" << printSignificantPattern(*sp) << "]";
    }
    else
        sout << *(nodes[node].lexicon);

    return sout.str();
}

string RDSGraph::printPath(const SearchPath &path) const
{
    ostringstream sout;

    sout << "[";
    for(unsigned int i = 0; i < path.size(); i++)
    {
        unsigned tempIndex = path[i];
        if(nodes[tempIndex].type == LexiconTypes::EC)
            sout << "E" << tempIndex;
        else if(nodes[tempIndex].type == LexiconTypes::SP)
            sout << "P" << tempIndex;
        else
            sout << *(nodes[tempIndex].lexicon);

        if(i < (path.size() - 1)) sout << " - ";
    }
    sout << "]";

    return sout.str();
}

std::string RDSGraph::printNodeName(unsigned int node) const
{
    ostringstream sout;

    if(nodes[node].type == LexiconTypes::EC)
        sout << "E" << node;
    else if(nodes[node].type == LexiconTypes::SP)
        sout << "P" << node;
    else
        sout << *(nodes[node].lexicon);

    return sout.str();
}

void printInfo(const ConnectionMatrix &connections, const Array2D<double> &flows, const Array2D<double> &descents)
{
    for(int i = 0; i < connections.dim1(); i++)
    {
        for(int j = 0; j < connections.dim2(); j++)
            std::cout << connections(i, j).size() << "\t";
        std::cout << endl;
    }
    std::cout << endl << endl << endl;
    for(int i = 0; i < flows.dim1(); i++)
    {
        for(int j = 0; j < flows.dim2(); j++)
            std::cout << flows(i, j) << "\t";
        std::cout << endl;
    }
    std::cout << endl << endl << endl;
    for(int i = 0; i < descents.dim1(); i++)
    {
        for(int j = 0; j < descents.dim2(); j++)
            std::cout << descents(i, j) << "\t";
        std::cout << endl;
    }
    std::cout << endl << endl << endl;
}
