#ifndef PARSE_TREE_H
#define PARSE_TREE_H

#include <vector>

typedef std::pair<unsigned int, unsigned int> Connection;

template <class T>
class ParseTree;

template <class T>
class ParseNode
{
    friend class ParseTree<T>;

    public:
        ParseNode()
        : the_parent(0, 0)
        {}

        ParseNode(const T &value, const Connection &parent)
        : the_value(value), the_parent(parent)
        {}

        const T& value() const
        {
            return the_value;
        }

        const std::vector<unsigned int>& children() const
        {
            return the_children;
        }

        std::vector<unsigned int> rewireChildren(unsigned int start, unsigned int finish, unsigned new_node)
        {
            std::vector<unsigned int> subsumed_part(the_children.begin()+start, the_children.begin()+finish+1);
            the_children.erase( the_children.begin()+start, the_children.begin()+finish+1);
            the_children.insert(the_children.begin()+start, new_node);
            return subsumed_part;

        }

    private:
        T the_value;
        Connection the_parent;
        std::vector<unsigned int> the_children;
};

template <class T>
class ParseTree
{
    public:
        ParseTree()
        {
            the_nodes.push_back(ParseNode<T>());   // nodes[0] is always the root
        }

        ParseTree(const std::vector<T> &values)
        {
            the_nodes.push_back(ParseNode<T>());   // nodes[0] is always the root
            for(unsigned int i = 0; i < values.size(); i++)
            {
                the_nodes.front().the_children.push_back(the_nodes.size());
                the_nodes.push_back(ParseNode<T>(values[i], Connection(0, i)));
            }
        }

        const std::vector<ParseNode<T> >& nodes() const
        {
            return the_nodes;
        }

        void rewire(unsigned int start, unsigned int finish, const T &new_node)
        {
            the_nodes.push_back(ParseNode<T>(new_node, Connection(0, 0)));
            the_nodes.back().the_children = the_nodes.front().rewireChildren(start, finish, the_nodes.size()-1);
            for(unsigned int i = 0; i < the_nodes.back().the_children.size(); i++)
                the_nodes[the_nodes.back().the_children[i]].the_parent = Connection(the_nodes.size()-1, i);
        }

        void attach(unsigned int attachPoint, const ParseTree<T> &branch)
        {
            assert(attachPoint < the_nodes.size());

            unsigned int offset = the_nodes.size();
            for(unsigned int i = 1; i < branch.the_nodes.size(); i++)
            {
                the_nodes.push_back(branch.the_nodes[i]);
                the_nodes.back().the_parent.first = the_nodes.back().the_parent.first+offset;
                for(unsigned int j = 0; j < the_nodes.back().children().size(); j++)
                    the_nodes.back().the_children[j] = the_nodes.back().the_children[j]+offset-1;
            }

            the_nodes[offset].the_parent.first = attachPoint;
            for(unsigned int i = 0; i < branch.the_nodes[0].the_children.size(); i++)
            the_nodes[attachPoint].the_children.push_back(branch.the_nodes[0].the_children[i]+offset-1);
        }

        void print(unsigned int node, unsigned int tab_level) const
        {
            for(unsigned int i = 0; i < tab_level; i++)
                std::cout << "\t";
            std::cout << node << " ---> " << the_nodes[node].the_value << std::endl;
            for(unsigned int i = 0; i < the_nodes[node].the_children.size(); i++)
                print(the_nodes[node].the_children[i], tab_level+1);
        }

    private:
        std::vector<unsigned int> the_leaves;
        std::vector<ParseNode<T> > the_nodes;
};

#endif
