from os import path, environ, makedirs
import spacy

SRC_ROOT_DIR = path.dirname(path.realpath(__file__))  # path to this script
ROOT_DIR = path.dirname(SRC_ROOT_DIR) # same as .. to move up one directory
DATA_DIR = path.join(ROOT_DIR, 'data')
OUTPUT_DIR = path.join(ROOT_DIR, 'output')

if not path.exists(DATA_DIR):
    makedirs(DATA_DIR)

if not path.exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)


nlp = spacy.load("it_core_news_sm")

# if __name__ == '__main__':
#     print(DATA_DIR)
#     print(SRC_ROOT_DIR)
#     print(OUTPUT_DIR)