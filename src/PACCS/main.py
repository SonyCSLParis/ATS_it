from open_data import *
from visualisations import *


def main():
    #final_corpus = open_txt()
    final_corpus = pd.read_csv('/Users/francesca/Desktop/Github/Final/output/output_modello/tts.csv')
    create_visualisation(final_corpus)


if __name__ == '__main__':
    main()
