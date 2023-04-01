from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np


class State(object):
    def __init__(self, sentence=[]):
        self.stack = []  # list of word ids
        self.buffer = []  # list of word ids
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()  # list of (parent, child, relation) triples

    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add((self.buffer[-1], self.stack.pop(), label))

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add((parent, self.buffer.pop(), label))
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)


def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label)
        elif rel == "right_arc":
            state.right_arc(label)

    return state.deps


class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None

    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_structure):
    # returns a list of (State, Transition) pairs
    # Transition is (transition, label)
    # dep_structure is  DependencyStructure
    deprels = dep_structure.deprels

    # get a sentence
    sorted_nodes = [k for k, v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident, node in deprels.items():
        childcount[node.head] += 1

    seq = []
    while state.buffer:
        if not state.stack:  # stack empty
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy()
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        # stackword and bufferword are DependencyEdge
        if stackword.head == bufferword.id:
            childcount[bufferword.id] -= 1
            seq.append((copy.deepcopy(state), ("left_arc", stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id] -= 1
            seq.append((copy.deepcopy(state), ("right_arc", bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else:
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
    return seq


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj',
                 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc',
                 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse',
                 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg',
                 'dt', 'det']


class FeatureExtractor(object):

    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)
        self.pos_vocab = self.read_vocab(pos_vocab_file)
        self.output_labels = self.make_output_labels()
        self.output_matrix = keras.utils.to_categorical(range(0,91))

    def make_output_labels(self):
        labels = []
        labels.append(('shift', None))

        for rel in dep_relations:
            labels.append(("left_arc", rel))
            labels.append(("right_arc", rel))
        return dict((label, index) for (index, label) in enumerate(labels))

    def read_vocab(self, vocab_file):
        vocab = {}
        for line in vocab_file:
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab

    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2
        input = np.zeros(6)

        def get_id(word_id):
            if word_id == 0:
                return self.word_vocab["<ROOT>"]
            elif pos[word_id] == "CD":
                return self.word_vocab["<CD>"]
            elif pos[word_id] == "NNP":
                return self.word_vocab["<NNP>"]
            elif words[word_id] not in self.word_vocab.keys():
                return self.word_vocab["<UNK>"]
            else:
                return self.word_vocab[words[word_id]]

        # stack
        length = 3 if len(state.stack) > 3 else len(state.stack)
        for i in range(length, 3):
            input[i] = self.word_vocab["<NULL>"]
        for i in range(0, length):
            input[i] = get_id(state.stack[-(i + 1)])
        # buffer
        length = 3 if len(state.buffer) > 3 else len(state.buffer)
        for i in range(length, 3):
            input[i + 3] = self.word_vocab["<NULL>"]
        for i in range(0, length):
            input[i + 3] = get_id(state.buffer[-(i + 1)])
        return input

    def get_output_representation(self, output_pair):
        # TODO: Write this method for Part 2
        id = self.output_labels[output_pair]
        return self.output_matrix[id]


def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0
    for dtree in conll_reader(in_file):
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs), np.vstack(outputs)


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    with open(sys.argv[1], 'r') as in_file:

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor, in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)

