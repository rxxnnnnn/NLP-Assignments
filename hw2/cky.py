"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg


### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and \
                isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str):  # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(
                        bps))
                return False
            if len(bps) != 2:
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(
                        bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp) != 3:
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(
                            bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(
                            bp))
                    return False
    return True


def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        n = len(tokens)
        table = defaultdict()
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                table[(i, j)] = defaultdict(list)
        # initialization
        for i in range(n):
            rules = self.grammar.rhs_to_rules[(tokens[i],)]
            for j in range(len(rules)):
                table[(i, i + 1)][rules[j][0]].append(tokens[i])
        # main loop
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    for B in table[(i, k)]:
                        for C in table[(k, j)]:
                            rules = self.grammar.rhs_to_rules[(B, C)]
                            for r in range(len(rules)):
                                table[(i, j)][rules[r][0]].append(((B, i, k), (C, k, j)))
        # check
        if self.grammar.startsymbol in table[(0, n)]:
            return True
        else:
            return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = defaultdict()
        probs = defaultdict()
        n = len(tokens)
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                table[(i, j)] = defaultdict()
                probs[(i, j)] = defaultdict(float)
        # initialization
        for i in range(n):
            rules = self.grammar.rhs_to_rules[(tokens[i],)]
            for j in range(len(rules)):
                table[(i, i + 1)][rules[j][0]] = tokens[i]
                probs[(i, i + 1)][rules[j][0]] = math.log(rules[j][2])
        # main loop
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    for B in table[(i, k)]:
                        for C in table[(k, j)]:
                            rules = self.grammar.rhs_to_rules[(B, C)]
                            for r in range(len(rules)):
                                if rules[r][0] not in table[(i, j)] or probs[(i, j)][rules[r][0]] < (
                                        math.log(rules[r][2]) + probs[(i, k)][B] + probs[(k, j)][C]):
                                    table[(i, j)][rules[r][0]] = ((B, i, k), (C, k, j))
                                    probs[(i, j)][rules[r][0]] = math.log(rules[r][2]) + probs[(i, k)][B] + \
                                                                 probs[(k, j)][C]
        return table, probs


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if isinstance(chart[(i, j)][nt], tuple):
        a = chart[(i, j)][nt][0][1]
        b = chart[(i, j)][nt][0][2]
        c = chart[(i, j)][nt][1][2]
        first = chart[(i, j)][nt][0][0]
        second = chart[(i, j)][nt][1][0]
        tree = nt, get_tree(chart, a, b, first), get_tree(chart, b, c, second)
        return tree
    else:
        return nt, chart[(i, j)][nt]


if __name__ == "__main__":
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        tree = get_tree(table, 0, len(toks), grammar.startsymbol)
        print(tree)
