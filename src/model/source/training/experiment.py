from transformers import (
    AutoTokenizer,
    EncoderDecoderModel, AutoModelForSeq2SeqLM)
from functools import lru_cache
import shutil
import Levenshtein
import numpy as np
from settings import *
FASTTEXT_EMBEDDINGS_PATH = '/Users/francesca/Desktop/wiki.it.vec'
import bz2
import gzip
import os
from pathlib import Path
import shutil
import sys
from string import punctuation
import spacy
import tarfile
import tempfile
import time
from urllib.request import urlretrieve
import zipfile

import git
from tqdm import tqdm
#from access.text import (to_words, remove_punctuation_tokens, remove_stopwords, spacy_process)
from pathlib import Path

print(len('semplifica: 0.83 0.8 0.7 1.0 1.0'))




'''def download_and_extract(url):
    tmp_dir = Path(tempfile.mkdtemp())
    compressed_filename = url.split('/')[-1]
    compressed_filepath = tmp_dir / compressed_filename
    download(url, compressed_filepath)
    print('Extracting...')
    return extract(compressed_filepath, tmp_dir)

@contextmanager
def lock_directory(dir_path):
    # TODO: Locking a directory should lock all files in that directory
    # Right now if we lock foo/, someone else can lock foo/bar.txt
    # TODO: Nested with lock_directory() should not be blocking
    assert Path(dir_path).exists(), f'Directory does not exists: {dir_path}'
    lockfile_path = get_lockfile_path(dir_path)
    with open_with_lock(lockfile_path, 'w'):
        yield'''


def prepare_fasttext_embeddings():
    FASTTEXT_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with lock_directory(FASTTEXT_EMBEDDINGS_PATH.parent):
        if FASTTEXT_EMBEDDINGS_PATH.exists():
            return
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz'
        extracted_path = download_and_extract(url)[0]
        shutil.move(extracted_path, FASTTEXT_EMBEDDINGS_PATH)

def count_lines(filepath):
    n_lines = 0
    with Path(filepath).open() as f:
        for l in f:
            n_lines += 1
    return n_lines


def yield_lines(filepath, n_lines=float('inf'), prop=1):
    if prop < 1:
        assert n_lines == float('inf')
        n_lines = int(prop * count_lines(filepath))
    with open(filepath, 'r') as f:
        for i, l in enumerate(f):
            if i >= n_lines:
                break
            yield l.rstrip('\n')

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


def to_words(sentence):
    return sentence.split()


def remove_punctuation_characters(text):
    return ''.join([char for char in text if char not in punctuation])

@lru_cache(maxsize=100)
def remove_punctuation_tokens(text):
    return ' '.join([w for w in to_words(text) if not is_punctuation(w)])

stopwords = nlp.Defaults.stop_words
def remove_stopwords(text):
    return ' '.join([w for w in to_words(text) if w.lower() not in stopwords])

@lru_cache(maxsize=1000)
def is_punctuation(word):
    return remove_punctuation_characters(word) == ''


def get_lexical_complexity_score(sentence):
    words = to_words(remove_stopwords(remove_punctuation_tokens(sentence)))
    words = [word for word in words if word in get_word2rank()]
    if len(words) == 0:
        return np.log(1 + len(get_word2rank()))  # TODO: This is completely arbitrary
    return np.quantile([get_log_rank(word) for word in words], 0.75)

def get_levenshtein_similarity(complex_sentence, simple_sentence):
    return Levenshtein.ratio(complex_sentence, simple_sentence)


@lru_cache(maxsize=1)
def get_spacy_model():
    model = 'it_core_news_sm'
    if not spacy.util.is_package(model):
        spacy.cli.download(model)
        spacy.cli.link(model, model, force=True, model_path=spacy.util.get_package_path(model))
    return spacy.load(model)


@lru_cache(maxsize=10**6)
def spacy_process(text):
    return get_spacy_model()(str(text))

def get_dependency_tree_depth(sentence):
    def get_subtree_depth(node):
        if len(list(node.children)) == 0:
            return 0
        return 1 + max([get_subtree_depth(child) for child in node.children])

    tree_depths = [get_subtree_depth(spacy_sentence.root) for spacy_sentence in spacy_process(sentence).sents]
    if len(tree_depths) == 0:
        return 0
    return max(tree_depths)



print(get_dependency_tree_depth('Mia nonna è caduta dalle scale e mia mamma è andata ad aiutarla mentre si rialzava'))