from os import path
import spacy

SRC_ROOT_DIR = path.dirname(path.realpath(__file__))  # path to this script

ROOT_DIR = path.dirname(SRC_ROOT_DIR)
print(ROOT_DIR)
DATA_DIR = path.join(ROOT_DIR + '/data')  # data in input

INTERMEDIATE_DIR = path.join(ROOT_DIR, '/intermediate')  # txt and others intermediate output
print(INTERMEDIATE_DIR)
OUTPUT_DIR = path.join(ROOT_DIR, '/output')  # final output
HTML_DIR = path.join(ROOT_DIR, '/html_output')  # plotly html save directory for final visualisations


