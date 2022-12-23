from src.data_merging.open_paccsit import *
from visualisations import *
from settings import *


def main(only_PACSSIT = None, dataset_name = None):

    if only_PACSSIT:
        final_corpus = open_txt()

    else:
        final_corpus = pd.read_csv(dataset_name)

    create_visualisation(final_corpus)


if __name__ == '__main__':
    main(only_PACSSIT = False, dataset_name= CSV_FILES_PATH + '/ter_tea_sim/tts.csv')
