from open_data import *
from visualisations import *


def main():
    final_corpus = open_txt()
    create_visualisation(final_corpus)


if __name__ == '__main__':
    print(DATA_DIR)
    print(SRC_ROOT_DIR)
    print(OUTPUT_DIR)
    main()
