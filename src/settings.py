from os import path, makedirs

import spacy

SRC_ROOT_DIR = path.dirname(path.realpath(__file__))  # path to this script
ROOT_DIR = path.dirname(SRC_ROOT_DIR)
DATA_DIR = path.join(ROOT_DIR, 'data')  # data in input
INTERMEDIATE_DIR = path.join(ROOT_DIR, 'intermediate')  # txt and others intermediate output
OUTPUT_DIR = path.join(ROOT_DIR, 'output')  # final output
MODEL_DIR = path.join(SRC_ROOT_DIR, 'model')  # directory for the model
HTML_DIR = path.join(ROOT_DIR, 'html_output')  # plotly html save directory for final visualisations

list_dirs = [DATA_DIR, INTERMEDIATE_DIR, OUTPUT_DIR, MODEL_DIR, HTML_DIR]

for x in list_dirs:
    if not path.exists(x):
        makedirs(x)

nlp = spacy.load("it_core_news_sm")
