import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np

"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    result = []
    if n > 1:
        sequence = ['START'] * (n - 1) + sequence + ['STOP']
    else:
        sequence = ['START'] + sequence + ['STOP']
    l = len(sequence)
    for i in range(l + 1 - n):
        result.append(tuple(sequence[i:i + n]))
    return result


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int)  # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.n_sentences = 0

        ##Your code here
        for sentence in corpus:
            self.n_sentences += 1
            uni = get_ngrams(sentence, 1)
            bi = get_ngrams(sentence, 2)
            tri = get_ngrams(sentence, 3)
            for u in uni:
                self.unigramcounts[u] += 1
            for b in bi:
                self.bigramcounts[b] += 1
            for t in tri:
                self.trigramcounts[t] += 1
        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[0:2] == ('START', 'START'):
            return self.trigramcounts[trigram] / self.n_sentences
        elif self.bigramcounts[trigram[0:2]] == 0:
            return 1/(len(self.lexicon)-1)
        else:
            return self.trigramcounts[trigram] / self.bigramcounts[trigram[0:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return self.bigramcounts[bigram] / self.unigramcounts[bigram[0:1]]

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        if not hasattr(self, 'uni_count'):
            self.uni_count = sum(self.unigramcounts.values()) - self.unigramcounts[('START',)]
        return self.unigramcounts[unigram] / self.uni_count

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = list()
        word_1 = "START"
        word_2 = "START"
        while word_2 != 'STOP' and len(result) < t:
            word_3_list = []
            prob = []
            for i in self.trigramcounts.keys():
                if i[0:2] == (word_1, word_2):
                    word_3_list.append(i[2])
                    prob.append(self.raw_trigram_probability(i))
            word_3 = np.random.choice(word_3_list, p=prob)
            result.append(word_3)
            word_1 = word_2
            word_2 = word_3

        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        return lambda1 * self.raw_trigram_probability(trigram) + \
                   lambda2 * self.raw_bigram_probability(trigram[1:3]) + \
                   lambda3 * self.raw_unigram_probability((trigram[2],))

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        sum = 0
        trigrams = get_ngrams(sentence, 3)
        for tri in trigrams:
            if self.smoothed_trigram_probability(tri) == 0:
                print(tri)
            sum += math.log2(self.smoothed_trigram_probability(tri))
        return sum

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum = 0
        M = 0
        for sentence in corpus:
            sum += self.sentence_logprob(sentence)
            M += (len(sentence) + 1)
        return 2 ** (-(sum / M))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if pp1 < pp2:
            correct += 1
        total += 1

    for f in os.listdir(testdir2):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        if pp1 > pp2:
            correct += 1
        total += 1


    return correct/total


if __name__ == "__main__":
    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)

    # Essay scoring experiment:
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    print(acc)
