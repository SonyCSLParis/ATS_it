# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import lru_cache
from settings import *
import Levenshtein
import numpy as np
from src.model.source.adaptive.text import (to_words, remove_punctuation_tokens, remove_stopwords, spacy_process)
from src.model.source.adaptive.helper import yield_lines

def safe_division(a, b):
    return a / b if b else 0

@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = nlp.tokenizer
    return tokenizer

def tokenize(sentence):
    tokenizer = nlp.tokenizer
    return tokenizer(sentence)


#this is the parameter which corresponds to #word
def get_word_length_ratio(complex_sentence, simple_sentence):
    return round(safe_division(len(tokenize(simple_sentence)), len(tokenize(complex_sentence))), 2)

#this is the parameter which corresponds to #chars
def get_char_length_ratio(complex_sentence, simple_sentence):
    return round(safe_division(len(simple_sentence), len(complex_sentence)), 2)

#this is the parameter that corresponds to the WordRank; it addresses the lexical complexity of our sentence
@lru_cache(maxsize=1)
def get_word2rank(vocab_size=np.inf):
    # TODO: Decrease vocab size or load from smaller file
    word2rank = {}
    line_generator = yield_lines(FASTTEXT_EMBEDDINGS_PATH)
    next(line_generator)  # Skip the first line (header)
    for i, line in enumerate(line_generator):
        if (i + 1) > vocab_size:
            break
        word = line.split(' ')[0]
        word2rank[word] = i
    return word2rank


def get_rank(word):
    return get_word2rank().get(word, len(get_word2rank()))


def get_log_rank(word):
    return np.log(1 + get_rank(word))

#first way to obtain the complexity score (quantile)
def get_lexical_complexity_score(sentence):
    words = to_words(remove_stopwords(remove_punctuation_tokens(sentence)))
    words = [word for word in words if word in get_word2rank()]
    if len(words) == 0:
        return np.log(1 + len(get_word2rank()))  # TODO: This is completely arbitrary
    return np.quantile([get_log_rank(word) for word in words], 0.75)

#method through which to obtain the normalized rank
@lru_cache(maxsize=10000)
def get_normalized_rank(word):
    max = len(get_word2rank())
    rank = get_word2rank().get(word, max)
    return np.log(1 + rank) / np.log(1 + max)
    # return np.log(1 + rank)

#second way to obtain the complexity score (mean) - we should get the normalized rank through the proper function
@lru_cache(maxsize=2048)
def get_complexity_score2(sentence):
    words = tokenize(remove_stopwords(remove_punctuation_tokens(sentence)))
    words = [word for word in words if word in get_word2rank()]  # remove unknown words
    if len(words) == 0:
        return 1.0
    return np.array([get_normalized_rank(word) for word in words]).mean()

#this is the parameter that corresponds to the amound of compression between the two sentences
def get_levenshtein_similarity(complex_sentence, simple_sentence):
    return Levenshtein.ratio(complex_sentence, simple_sentence)

#this method let you manage to calculate the tree depth
def get_dependency_tree_depth(sentence):
    def get_subtree_depth(node):
        if len(list(node.children)) == 0:
            return 0
        return 1 + max([get_subtree_depth(child) for child in node.children])

    tree_depths = [get_subtree_depth(spacy_sentence.root) for spacy_sentence in spacy_process(sentence).sents]
    if len(tree_depths) == 0:
        return 0
    return max(tree_depths)


#this parameter accounts for the syntactic complexity of the sentences
def get_dependency_tree_depth_ratio(complex_sentence, simple_sentence):
    return round(safe_division(get_dependency_tree_depth(simple_sentence),
                      get_dependency_tree_depth(complex_sentence)))





