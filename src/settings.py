from os import path
import spacy

SRC_ROOT_DIR = path.dirname(path.realpath(__file__))  # path to this script
SOURCE_DIR = path.join(SRC_ROOT_DIR + '/src') # path to the src script
DATA_DIR = path.join(SRC_ROOT_DIR + '/data')  # data (datasets) in input
INTERMEDIATE_DIR = path.join(SRC_ROOT_DIR + '/intermediate')  # txt and others intermediate output
INCOMPLETE_DATASET_DIR = path.join(INTERMEDIATE_DIR + '/incomplete_datasets') # primarily processed datasets
OUTPUT_DIR = path.join(SRC_ROOT_DIR + '/output')  # datasets in output
HF_DATASETS = path.join(OUTPUT_DIR + '/hugging_face')
CSV_FILES_PATH = path.join(OUTPUT_DIR + '/csv_files')
TOKENIZER_PATH = "dbmdz/bert-base-italian-xxl-uncased"


MODEL_DIR_GENERAL = path.join(SOURCE_DIR + '/model') #general model directory
MODEL_DEEP_MARTIN = path.join(MODEL_DIR_GENERAL + '/deep_mart_final') #Deep Martin Transformer path
SOURCE_MODEL_DIR = path.join(MODEL_DEEP_MARTIN + '/source')
BERT2BERT_DIR = SOURCE_MODEL_DIR + '/bert2bert'

TRAINED_MODEL = path.join(MODEL_DEEP_MARTIN + '/model_deep' + '/trained_model') #trained models checkpoints
CSV_EVAL_OUTPUT = path.join(MODEL_DEEP_MARTIN + '/model_deep' + '/csv_output') #output csv after evaluation

HTML_DIR = path.join(SRC_ROOT_DIR +'/html_output')  # plotly html save directory for final visualisations


nlp = spacy.load('it_core_news_sm')
