from open_data import *
from visualisations import *


def main(only_PACSSIT = None, dataset_name = None):
    if only_PACSSIT:
        final_corpus = open_txt()

    else:
        final_corpus = pd.read_csv(dataset_name)
        create_visualisation(final_corpus)


if __name__ == '__main__':
    main(only_PACSSIT = True, dataset_name= None)
