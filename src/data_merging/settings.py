from os import path, makedirs

import spacy

SRC_ROOT_DIR = '/Users/francesca/PycharmProjects/ATS_Sony/ATS_it' # path to this script
ROOT_DIR = path.dirname(SRC_ROOT_DIR)
DATA_DIR = path.join(ROOT_DIR, 'data')  # data in input

INTERMEDIATE_DIR = path.join(ROOT_DIR, 'intermediate')  # txt and others intermediate output
INCOMPLETE_DATASET_DIR = path.join(INTERMEDIATE_DIR + 'incomplete_datasets')
OUTPUT_DIR = path.join(ROOT_DIR, 'output')  # final output
#MODEL_DIR = path.join(SRC_ROOT_DIR, '../model_implementation')  # directory for the model_deep
HTML_DIR = path.join(ROOT_DIR, 'html_output')  # plotly html save directory for final visualisations


nlp = spacy.load("it_core_news_sm")