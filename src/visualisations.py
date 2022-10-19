from PoS import PoS_analysis, barplot_PoS
from settings import *
from stopwords_distribution import amount_words, stopwords, plot_stop
from verb_mood_tense import verbal_features, double_barplot, coord_subord

def create_visualisation(final_corpus):
    which_stop, average_amount = stopwords(2, final_corpu=final_corpus, filter=False, save=False)
    print(average_amount)
    print(amount_words(2, final_corpu=final_corpus))
    plot_stop(stopwords=which_stop, filter='filtered', original='simplified', color='olive')
    key1, value1 = PoS_analysis(OUTPUT_DIR + '/no_stop_sentences1.txt')
    key2, value2 = PoS_analysis(OUTPUT_DIR + '/no_stop_sentences2.txt')
    barplot_PoS(key1, value1, value2)

    mood_complex = verbal_features(typev='Mood',
                                   filename=OUTPUT_DIR + '/no_stop_sentences1.txt')
    mood_simple = verbal_features(typev='Mood',
                                  filename=OUTPUT_DIR + '/no_stop_sentences2.txt')
    tense_complex = verbal_features(typev='Tense',
                                    filename=OUTPUT_DIR + '/no_stop_sentences1.txt')
    tense_simple = verbal_features(typev='Tense',
                                   filename=OUTPUT_DIR + '/no_stop_sentences2.txt')

    double_barplot(mood_complex, mood_simple, 'indianred', 'lightsalmon')
    double_barplot(tense_complex, tense_simple, 'dodgerblue', 'skyblue')

    # Calculating the amount of coordination conjunctions are present in the complex and simple text
    tot_complex, collection_complex = coord_subord(final_corpus, 1, 'CCONJ')
    tot_simple, collection_simple = coord_subord(final_corpus, 2, 'CCONJ')
    tot_simple1, collection_simple1 = coord_subord(final_corpus, 2, 'SCONJ')
    print(tot_complex, tot_simple, tot_simple1)

    print('The end!')