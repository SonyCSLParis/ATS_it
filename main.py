from stopwords_distribution import is_number, amount_words, stopwords, plot_stop
from PoS import PoS_analysis, barplot_PoS
from verb_mood_tense import verbal_features, double_barplot, coord_subord
import csv
import pandas as pd


def main():
    #Open the corpus and start parsing it
    with open('/Users/francesca/Desktop/ARTIS/DATASET/PACCSS-IT.txt', "r") as infile:
        reader = csv.reader(infile, delimiter='\t')
        header = next(reader)  # skip header

        length = [7]
        count = 0
        corpus = []
        for line in reader:
            lunghezza = len(line)
            if lunghezza not in length:
                count += 1
            else:
                corpus.append(line)

    #Some lines (rows) where not properly designed and build, therefore we discarded them and print how many of them were not considered
    print(f'The wrong lines are: {count}.')

    #Reassign the column names (header)
    final_corpus = pd.DataFrame(corpus)
    final_corpus = final_corpus.rename(
        {0: 'Sentence_1', 1: 'Sentence_2', 2: 'Cosine_Similarity', 3: 'Confidence', 4: 'Readability_1',
         5: 'Readability_2',
         6: '(Readability_1-Readability_2)'}, axis=1)

    which_stop, average_amount = stopwords(2, final_corpu=final_corpus, filter=False, save = False)
    print(average_amount)
    print(amount_words(2, final_corpu=final_corpus))
    plot_stop(stopwords=which_stop, filter='filtered', original='simplified', color='olive')
    key1, value1 = PoS_analysis('/Users/francesca/PycharmProjects/pythonProject/no_stop_sentences1.txt')
    key2, value2 = PoS_analysis('/Users/francesca/PycharmProjects/pythonProject/no_stop_sentences2.txt')
    barplot_PoS(key1, value1, value2)

    mood_complex = verbal_features(typev='Mood',
                                   filename='/Users/francesca/PycharmProjects/pythonProject/no_stop_sentences1.txt')
    mood_simple = verbal_features(typev='Mood',
                                  filename='/Users/francesca/PycharmProjects/pythonProject/no_stop_sentences2.txt')
    tense_complex = verbal_features(typev='Tense',
                                    filename='/Users/francesca/PycharmProjects/pythonProject/no_stop_sentences1.txt')
    tense_simple = verbal_features(typev='Tense',
                                   filename='/Users/francesca/PycharmProjects/pythonProject/no_stop_sentences2.txt')

    double_barplot(mood_complex, mood_simple)
    double_barplot(tense_complex, tense_simple)

    #Calculating the amount of coordination conjunctions are present in the complex and simple text
    tot_complex, collection_complex = coord_subord(final_corpus, 1, 'CCONJ')
    tot_simple, collection_simple = coord_subord(final_corpus, 2, 'CCONJ')
    tot_simple1, collection_simple1 = coord_subord(final_corpus, 2, 'SCONJ')
    print(tot_complex, tot_simple, tot_simple1)

    print('Fine!')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

