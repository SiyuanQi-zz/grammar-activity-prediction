"""
Created on Feb 24, 2017

@author: Siyuan Qi

Description of the file.

"""

import collections
import os
import time
import itertools

import numpy as np
import nltk

import config


def induce_activity_grammar(paths):
    """
    Parameters for grammar induction:
    <eta>:          threshold of detecting divergence in the RDS graph, usually set to 0.9
    <alpha>:        significance test threshold, usually set to 0.01 or 0.1
    <context_size>: size of the context window used for search for Equivalence Class, usually set to 5 or 4, a value
                    less than 3 means that no equivalence class can be found.
    <coverage>:     threhold for bootstrapping Equivalence classes, usually set to 0.65. Higher values will result in
                    less bootstrapping.

    :param paths: paths configuration of the project
    :return:
    """
    # TODO: experiment on the parameters
    adios_path = os.path.join(paths.project_root, 'src', 'cpp', 'madios', 'madios')
    eta = 1
    alpha = 0.1
    context_size = 4
    coverage = 0.5

    for corpus_filename in os.listdir(os.path.join(paths.tmp_root, 'corpus')):
        corpus_path = os.path.join(paths.tmp_root, 'corpus', corpus_filename)
        os.system('{} {} {} {} {} {}'.format(adios_path, corpus_path, eta, alpha, context_size, coverage))


def read_languages(paths):
    languages = dict()
    for corpus_filename in os.listdir(os.path.join(paths.tmp_root, 'corpus')):
        corpus_path = os.path.join(paths.tmp_root, 'corpus', corpus_filename)
        with open(corpus_path) as f:
            language = f.readlines()
            language = [s.strip().split() for s in language]
            language = [s[1:-1] for s in language]
            languages[os.path.splitext(corpus_filename)[0]] = language
    return languages


def get_pcfg(rules):
    root_rules = list()
    non_terminal_rules = list()
    grammar_rules = list()
    for rule in rules:
        tokens = rule.split()
        for i in range(len(tokens)):
            token = tokens[i]
            if token[0] == 'E':
                tokens[i] = tokens[i].replace('E', 'OR')
            elif token[0] == 'P':
                tokens[i] = tokens[i].replace('P', 'AND')
        rule = ' '.join(tokens)

        if rule.startswith('S'):
            root_rules.append(rule)
        else:
            non_terminal_rules.append(rule)

    for k, v in collections.Counter(root_rules).items():
        grammar_rules.append(k + ' [{}]'.format(float(v) / len(root_rules)))
    grammar_rules.extend(non_terminal_rules)
    return grammar_rules


def read_induced_grammar(paths):
    # Read grammar into nltk
    grammar_dict = dict()
    for activity_grammar_file in os.listdir(os.path.join(paths.tmp_root, 'grammar')):
        with open(os.path.join(paths.tmp_root, 'grammar', activity_grammar_file)) as f:
            rules = [rule.strip() for rule in f.readlines()]
            grammar_rules = get_pcfg(rules)
            grammar = nltk.PCFG.fromstring(grammar_rules)
            grammar_dict[os.path.splitext(activity_grammar_file)[0]] = grammar
            # print activity_grammar_file
            # print grammar
    return grammar_dict


def get_production_prob(selected_edge, grammar):
    # Find the corresponding production rule of the edge, and return its probability
    for production in grammar.productions(lhs=selected_edge.lhs()):
        if production.rhs() == selected_edge.rhs():
            # print selected_edge, production.prob()
            return production.prob()


def find_parent(selected_edge, chart):
    # Find the parent edges that lead to the selected edge
    p_edges = list()
    for p_edge in chart.edges():
        # Important: Note that p_edge.end() is not equal to p_edge.start() + p_edge.dot(),
        # when a node in the edge spans several tokens in the sentence
        if p_edge.end() == selected_edge.start() and p_edge.nextsym() == selected_edge.lhs():
            p_edges.append(p_edge)
    return p_edges


def get_edge_prob(selected_edge, chart, grammar):
    # Compute the probability of the edge by recursion
    prob = get_production_prob(selected_edge, grammar)
    if selected_edge.start() != 0:
        parent_prob = 0
        for parent_edge in find_parent(selected_edge, chart):
            parent_prob += get_edge_prob(parent_edge, chart, grammar)
        prob *= parent_prob
    return prob


def remove_duplicate(tokens):
    return [t[0] for t in itertools.groupby(tokens)]


def predict_next_symbols(grammar, tokens):
    tokens = remove_duplicate(tokens)
    symbols = list()
    earley_parser = nltk.EarleyChartParser(grammar, trace=0)
    try:
        e_chart = earley_parser.chart_parse(tokens)
    except ValueError:
        return list()
    end_edges = list()

    for edge in e_chart.edges():
        # print edge
        if edge.end() == len(tokens):
            # Only add terminal nodes
            if isinstance(edge.nextsym(), unicode):
                symbols.append(edge.nextsym())
                end_edges.append(edge)

    probs = list()
    for end_edge in end_edges:
        probs.append(get_edge_prob(end_edge, e_chart, grammar))

    # Eliminate duplicate
    symbols_no_duplicate = list()
    probs_no_duplicate = list()
    for s, p in zip(symbols, probs):
        if s not in symbols_no_duplicate:
            symbols_no_duplicate.append(s)
            probs_no_duplicate.append(p)
        else:
            probs_no_duplicate[symbols_no_duplicate.index(s)] += p

    return symbols_no_duplicate, probs_no_duplicate


def lcs(valid_tokens, tokens):
    lengths = [[0 for j in range(len(tokens) + 1)] for i in range(len(valid_tokens) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(valid_tokens):
        for j, y in enumerate(tokens):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

    x, y = len(valid_tokens), len(tokens)
    matched_tokens = None
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert valid_tokens[x-1] == tokens[y-1]
            matched_tokens = valid_tokens[:x]
            for i in range(x, len(tokens)):
                if i < len(valid_tokens):
                    matched_tokens.append(valid_tokens[i])
                else:
                    matched_tokens.append(valid_tokens[-1])
            break

    if not matched_tokens:
        if len(valid_tokens) < len(tokens):
            matched_tokens = valid_tokens[:]
            for _ in range(len(valid_tokens), len(tokens)):
                matched_tokens.append(valid_tokens[-1])
        else:
            matched_tokens = valid_tokens[:len(tokens)]
    return len(tokens) - lengths[-1][-1], matched_tokens


def find_closest_tokens(language, tokens, truncate=False):
    min_distance = np.inf
    best_matched_tokens = None

    for valid_tokens in language:
        d, matched_tokens = lcs(valid_tokens, tokens)
        if d < min_distance:
            min_distance = d
            if not truncate:
                best_matched_tokens = matched_tokens
            else:
                best_matched_tokens = matched_tokens[:len(valid_tokens)]

    return min_distance, best_matched_tokens


def compute_sentence_probability(grammar, language, tokens):
    invalid_prob = 1e-20

    earley_parser = nltk.EarleyChartParser(grammar, trace=0)
    viterbi_parser = nltk.ViterbiParser(grammar)
    try:
        e_chart = earley_parser.chart_parse(tokens)
    except ValueError:
        return 0
        # d, tokens = find_closest_tokens(language, tokens)
        # return invalid_prob ** d

    # If the sentence is complete, return the Viterbi likelihood
    v_parses = viterbi_parser.parse_all(tokens)
    if v_parses:
        prob = reduce(lambda a, b: a+b.prob(), v_parses, 0)/len(v_parses)
        return prob

    # If the sentence is incomplete, return the sum of probabilities of all possible sentences
    prob = 0
    for edge in e_chart.edges():
        if edge.end() == len(tokens) and isinstance(edge.nextsym(), unicode):
            prob += get_edge_prob(edge, e_chart, grammar)
    return prob


def sample_complete_sentence(grammar, tokens):
    # tokens should not exceed the longest possible sentence from the grammar
    # Set truncate=True when calling find_closest_tokens() to find a matched sentence
    complete_tokens = tokens[:]
    viterbi_parser = nltk.ViterbiParser(grammar)
    while not viterbi_parser.parse_all(complete_tokens):
        symbols, probs = predict_next_symbols(grammar, complete_tokens)
        try:
            complete_tokens.append(symbols[np.argmax(probs)])
        except ValueError:
            # Cannot predict the next token (symbols and probs are empty lists), but the sentence is incomplete
            print tokens
            print complete_tokens
    return complete_tokens


def get_prediciton_parse_tree(grammar, tokens, filename=None):
    complete_tokens = sample_complete_sentence(grammar, tokens)

    earley_parser = nltk.EarleyChartParser(grammar, trace=0)
    e_chart = earley_parser.chart_parse(complete_tokens)

    # Select the first parse tree
    parse_tree = e_chart.parses(grammar.start()).next()

    # Save the parse tree as an image file if a filename is given
    if filename:
        cf = nltk.draw.util.CanvasFrame()
        tc = nltk.draw.TreeWidget(cf.canvas(), parse_tree)

        # Customize your own graph
        tc['node_font'] = 'arial 45 bold'
        tc['leaf_font'] = 'arial 45'
        tc['node_color'] = '#005990'
        tc['leaf_color'] = '#3F8F57'
        tc['line_color'] = '#175252'
        tc['line_width'] = '5'
        tc['xspace'] = 20
        tc['yspace'] = 20

        # Set color for the past observations.
        # Note that tc._leaves has more nodes than the leaves of the tree, and the last ones are displayed.
        for i in range(len(tc._leaves)-len(complete_tokens), len(tc._leaves)-len(complete_tokens)+len(tokens)-1):
            tc._leaves[i]['color'] = '#000000'
        # Set color for the current observation
        tc._leaves[len(tc._leaves)-len(complete_tokens)+len(tokens)-1]['color'] = '#FF0000'

        cf.add_widget(tc, 10, 10)  # (10,10) offsets
        # cf.mainloop()
        cf.print_to_file(filename)
        cf.destroy()
        basename, ext = os.path.splitext(filename)
        os.system('convert {} {}'.format(filename, basename+'.png'))
        os.remove(filename)

    return parse_tree


def grammar_to_dot(grammar, filename):
    and_nodes = list()
    or_nodes = list()
    terminal_nodes = list()
    root_branch_count = 0

    edges = list()
    for production in grammar.productions():
        if production.prob() == 1:
            and_nodes.append(str(production.lhs()))
            for i, child_node in enumerate(production.rhs()):
                edges.append(str(production.lhs()) + ' -> ' + str(child_node) + u' [penwidth=3, weight=3, label={}]\n'.format(unichr(9312+i)))
        else:
            or_nodes.append(str(production.lhs()))
            if str(production.lhs()) == 'S':
                root_branch_count += 1
                and_nodes.append('S{}'.format(root_branch_count))
                edges.append(str(production.lhs()) + ' -> ' + str('S{}'.format(root_branch_count)) + '[label = "' + "{0:.2f}".format(production.prob()) + '", penwidth=' + str(1. + 2.*production.prob()) + ', weight=3]\n')
                for i, child_node in enumerate(production.rhs()):
                    edges.append(str('S{}'.format(root_branch_count)) + ' -> ' + str(child_node) + u' [penwidth=3, weight=3, label={}]\n'.format(unichr(9312+i)))
            else:
                for child_node in production.rhs():
                    edges.append(str(production.lhs()) + ' -> ' + str(child_node) + '[label = "' + "{0:.2f}".format(production.prob()) + '", penwidth=' + str(1. + 2.*production.prob()) + ', weight=3]\n')

        for child_node in production.rhs():
            if isinstance(child_node, unicode):
                terminal_nodes.append(child_node)

    vertices = list()
    and_nodes = set(and_nodes)
    or_nodes = set(or_nodes)
    terminal_nodes = set(terminal_nodes)

    for and_node in and_nodes:
        vertices.append(and_node + ' [shape=doublecircle, fillcolor=green, style=filled, color=blue, ranksep=0.5, nodesep=0.5]\n')
    for or_node in or_nodes:
        vertices.append(or_node + ' [shape=circle, fillcolor=yellow, style=filled, color=blue, ranksep=0.5, nodesep=0.5]\n')
    for terminal in terminal_nodes:
        vertices.append(terminal + ' [shape=box, fillcolor=white, style=filled, ranksep=0.5, nodesep=0.5]\n')

    # edges = set(edges)
    with open(filename, 'w') as f:
        f.write('digraph G {\nordering=out\n')

        for vertex in vertices:
            f.write(vertex)

        for edge in edges:
            print edge
            f.write(edge.encode('utf8'))

        f.write('}')


def test(paths):
    grammar_dict = read_induced_grammar(paths)
    languages = read_languages(paths)

    #########################################
    # # Draw grammar
    # fig_folder = os.path.join(paths.tmp_root, 'figs', 'grammar')
    # ext = 'pdf'
    # if not os.path.exists(fig_folder):
    #     os.makedirs(fig_folder)
    # for task, grammar in grammar_dict.items():
    #     grammar = grammar_dict[task]
    #     dot_filename = os.path.join(fig_folder, '{}.dot'.format(task))
    #     fig_filename = os.path.join(fig_folder, '{}.{}'.format(task, ext))
    #     grammar_to_dot(grammar, dot_filename)
    #     os.system('dot -T{} {} > {}'.format(ext, dot_filename, fig_filename))
    #     # break

    # sentence = 'null reaching opening reaching moving cleaning moving placing reaching closing'

    # grammar = grammar_dict['stacking_objects']
    # sentence = 'null reaching'

    # grammar = grammar_dict['unstacking_objects']
    # language = languages['unstacking_objects']
    # sentence = 'null reaching moving placing reaching moving placing reaching moving placing'
    # sentence = 'null reaching eating placing reaching moving placing reaching moving playing'

    # tokens = sentence.split()
    # d, matched_tokens = find_closest_tokens(language, tokens)
    # print d, matched_tokens
    # print compute_sentence_probability(grammar, language, tokens)
    # print predict_next_symbols(grammar, matched_tokens)

    # sentence = 'null reaching moving placing reaching moving placing reaching movin'
    # tokens = sentence.split()
    # print tokens
    # for activity, grammar in grammar_dict.items():
    #     language = languages[activity]
    #     d, matched_tokens = find_closest_tokens(language, tokens)
    #     print activity, d, matched_tokens
    #     print compute_sentence_probability(grammar, language, tokens)
    #     print predict_next_symbols(grammar, matched_tokens)

    #########################################
    # Draw prediction parse graph
    grammar = grammar_dict['stacking_objects']
    sentence = 'null reaching'
    # sentence = 'null reaching moving placing reaching moving placing null reaching moving placing null'
    # sentence = 'null reaching moving placing reaching moving placing'
    tokens = sentence.split()
    # print predict_next_symbols(grammar, tokens)
    parse_tree = get_prediciton_parse_tree(grammar, tokens, 'tree.ps')


def main():
    paths = config.Paths()
    start_time = time.time()
    # induce_activity_grammar(paths)
    # read_induced_grammar(paths)
    test(paths)
    print('Time elapsed: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
