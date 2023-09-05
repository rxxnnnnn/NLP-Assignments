#!/usr/bin/env python
import string
import sys
from collections import defaultdict

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers

from typing import List


def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()


def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    ll = wn.lemmas(lemma.replace(' ', '_'), pos=pos)
    result = []
    for l in ll:
        ss = l.synset().lemmas()
        for s in ss:
            if s.name() != lemma and s.name() not in result:
                result.append(s.name().replace('_', ' '))
    return result


def smurf_predictor(context: Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context: Context) -> str:
    ll = wn.lemmas(context.lemma.replace(' ', '_'), pos=context.pos)
    result = defaultdict(int)
    for l in ll:
        ss = l.synset().lemmas()
        for s in ss:
            if s.name() != context.lemma.replace(' ', '_'):
                result[s.name()] += s.count()
    return max(result, key=result.get).replace('_', ' ')  # replace for part 2
'''
Total = 298, attempted = 298
precision = 0.098, recall = 0.098
Total with mode 206 attempted 206
precision = 0.136, recall = 0.136
'''

def wn_simple_lesk_predictor(context: Context) -> str:
    stop_words = stopwords.words('english')
    context_words = context.left_context + context.right_context
    context_words = [w.lower() for w in context_words]
    for w in context_words:
        if w in stop_words or w in string.punctuation:
            context_words.remove(w)
    context_words = set(context_words)
    ll = wn.lemmas(context.lemma.replace(' ', '_'), pos=context.pos)
    count = defaultdict(int)
    for i, l in enumerate(ll):
        ss = l.synset()
        def_ex = [ss.definition()] + ss.examples()
        for h in ss.hypernyms():
            def_ex += [h.definition()]
            def_ex += h.examples()
        def_ex_words = []
        for s in def_ex:
            def_ex_words += tokenize(s)
        def_ex_words = [w.lower() for w in def_ex_words]
        for w in def_ex_words:
            if w in stop_words or w in string.punctuation:
                def_ex_words.remove(w)
        def_ex_words = set(def_ex_words)
        overlap = len(def_ex_words & context_words)
        for s in ss.lemmas():
            if s.name() != context.lemma.replace(' ', '_'):
                count[(i, s.name())] = 80000 * overlap + 400 * l.count() + s.count()
    return max(count, key=count.get)[1].replace('_', ' ')  # replace for part 3
'''
Total = 298, attempted = 298
precision = 0.086, recall = 0.086
Total with mode 206 attempted 206
precision = 0.121, recall = 0.121
'''


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self, context: Context) -> str:
        synonyms = get_candidates(context.lemma, context.pos)
        max_sim = 0
        output = ""
        for s in synonyms:
            if s in self.model.key_to_index:
                sim = self.model.similarity(context.lemma, s)
                if sim > max_sim:
                    max_sim = sim
                    output = s
        return output  # replace for part 4
    '''
    Total = 298, attempted = 298
    precision = 0.115, recall = 0.115
    Total with mode 206 attempted 206
    precision = 0.170, recall = 0.170
    '''
    # Part6: consider the context
    def predict_nearest_improved(self, context: Context) -> str:
        synonyms = get_candidates(context.lemma, context.pos)
        max_sim = 0
        output = ""
        left = ""
        right = ""
        if len(context.left_context)>=1:
            left = context.left_context[-1]
        if len(context.right_context)>=1:
            right = context.right_context[0]
        for s in synonyms:
            if s in self.model.key_to_index:
                sim = 10 * self.model.similarity(context.lemma, s)
                if left in self.model.key_to_index:
                    sim += self.model.similarity(left, s)
                if right in self.model.key_to_index:
                    sim += self.model.similarity(right, s)
                if sim > max_sim:
                    max_sim = sim
                    output = s
        return output


class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        synonyms = get_candidates(context.lemma, context.pos)
        input = ' '.join(context.left_context + ['[MASK]'] + context.right_context)
        input_toks = self.tokenizer.encode(input)
        input_convert = self.tokenizer.convert_ids_to_tokens(input_toks)
        target_index = input_convert.index('[MASK]')
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat, verbose=False)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][target_index])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_words)

        for i, w in enumerate(best_words):
            if w.replace('_', ' ') in synonyms:
                return w.replace('_', ' ')

        return best_words[0].replace('_', ' ') # replace for part 5
    '''
    Total = 298, attempted = 298
    precision = 0.115, recall = 0.115
    Total with mode 206 attempted 206
    precision = 0.170, recall = 0.170
    '''

if __name__ == "__main__":
    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    #predictor = BertPredictor()
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.predict_nearest_improved(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
