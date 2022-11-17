from os import path, makedirs

SRC_ROOT_DIR = path.dirname(path.realpath(__file__))  # path to this script
ROOT_DIR = path.dirname(SRC_ROOT_DIR)
SRC_DIR = path.join(SRC_ROOT_DIR, 'source')  # data in input
ANALYSIS_DIR = SRC_DIR + '/analysis'
BERT2BERT_DIR = SRC_DIR + '/bert2bert'
print(BERT2BERT_DIR)
CT_DIR = SRC_DIR + '/custom_transformer/model_implementation'

MODEL_DIR = path.join(SRC_ROOT_DIR, '/model_deep')  # directory for the model_deep
TRAINED_MODELS_DIR = path.join(MODEL_DIR, '/trained_model')

IMAGES_DIR = path.join(SRC_ROOT_DIR, '/images')  # plotly html save directory for final visualisations
LOSS_PLOT_DIR = path.join(IMAGES_DIR + '/loss_plot')

list_dirs = [BERT2BERT_DIR,CT_DIR]

for x in list_dirs:
    if not path.exists(x):
        makedirs(x)