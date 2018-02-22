#include "RDSNode.h"

using std::vector;

RDSNode::RDSNode()
{
    this->lexicon = 0;
}

RDSNode::RDSNode(LexiconUnit *lexicon, LexiconTypes::LexiconEnum type)
{
    this->lexicon = lexicon;
    this->type = type;
}

RDSNode::RDSNode(const RDSNode &other)
{
    deepCopy(other);
}

RDSNode::~RDSNode()
{
    if(lexicon) delete lexicon;
}

RDSNode& RDSNode::operator=(const RDSNode &other)
{
    deepCopy(other);
    return *this;
}

void RDSNode::addConnection(const Connection &con)
{
    connections.push_back(con);
}

const vector<Connection>& RDSNode::getConnections() const
{
    return connections;
}

void RDSNode::setConnections(const vector<Connection> &connections)
{
    this->connections = connections;
}

bool RDSNode::addParent(const Connection &newParent)
{
    for(unsigned int i = 0; i < parents.size(); i++)
        if(parents[i] == newParent)
            return false;

    parents.push_back(newParent);
    return true;
}

void RDSNode::deepCopy(const RDSNode &other)
{
    lexicon = other.lexicon->makeCopy();
    type = other.type;
    connections = other.connections;
    parents = other.parents;
}
