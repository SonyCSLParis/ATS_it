from os import path
import spacy

SRC_ROOT_DIR = path.dirname(path.realpath(__file__))  # path to this script
DATA_DIR = path.join(SRC_ROOT_DIR + '/input')  # data (datasets) in input

ENGLISH_CORPUS = path.join(SRC_ROOT_DIR + '/english_corpus') #data for augmentation
TURKUS_DIR = path.join(ENGLISH_CORPUS + '/turkus')
TURKUS_TRANSLATED = path.join(TURKUS_DIR + '/translated')
TURKUS_COS_SIM = path.join(TURKUS_DIR + '/similarity')
WIKIPEDIA_DIR = path.join(ENGLISH_CORPUS + '/wikipedia')
WIKIPEDIA_TRANSLATED = path.join(WIKIPEDIA_DIR + '/translated_training')

SOURCE_DIR = path.join(SRC_ROOT_DIR + '/src') # path to the src script
INTERMEDIATE_DIR = path.join(SRC_ROOT_DIR + '/intermediate')  # txt and others intermediate output
INCOMPLETE_DATASET_DIR = path.join(INTERMEDIATE_DIR + '/incomplete_datasets') # primarily processed datasets
INCOMPLETE_NO_PROCESSED = path.join(INCOMPLETE_DATASET_DIR + '/not_processed')
INCOMPLETE_PROCESSED = path.join(INCOMPLETE_DATASET_DIR + '/processed')
OUTPUT_DIR = path.join(SRC_ROOT_DIR + '/output')  # datasets in output
HF_DATASETS = path.join(OUTPUT_DIR + '/hugging_face')
CSV_FILES_PATH = path.join(OUTPUT_DIR + '/csv_files')
TOKENIZER_PATH = "dbmdz/bert-base-italian-xxl-cased"


MODEL_DIR_GENERAL = path.join(SOURCE_DIR + '/model') #general model directory
SOURCE_MODEL_DIR = path.join(MODEL_DIR_GENERAL + '/source')
BERT2BERT_DIR = SOURCE_MODEL_DIR + '/bert2bert'
BERT2BERT_CASED_DIR = SOURCE_MODEL_DIR + '/bert2bert_cased'

TRAINED_MODEL = path.join(MODEL_DIR_GENERAL + '/model_deep' + '/trained_model') #trained models checkpoints
CSV_EVAL_OUTPUT = path.join(MODEL_DIR_GENERAL + '/model_deep' + '/csv_output') #output csv after evaluation

HTML_DIR = path.join(SRC_ROOT_DIR +'/html_output')  # plotly html save directory for final visualisations
AUTHENTICATION_DEEPL = 'b95b8cb8-cb6b-c36a-e49e-d17dac11061d:fx'
FASTTEXT_EMBEDDINGS_PATH = '/Users/francesca/Desktop/Github/PROJECT_SONY/input/embedding/wiki.it.vec'

nlp = spacy.load('it_core_news_sm')
