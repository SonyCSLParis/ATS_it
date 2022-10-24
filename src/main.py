from open_data import *
from visualisations import *


def main():
    final_corpus = open_txt()
    print(final_corpus)
    create_visualisation(final_corpus)


if __name__ == '__main__':
    main()
