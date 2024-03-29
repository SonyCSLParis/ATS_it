from verb_mood_tense import verbal_features, double_barplot, coord_subord, amount_SVO
from stopwords_distribution import *
from PoS import PoS_analysis, barplot_PoS

def create_visualisation(final_corpus):
    dimension = len(final_corpus)
    # average amount of words per sentences
    print('AVERAGE AMOUNT OF WORDS - complex ', amount_words(1, final_corpu=final_corpus, corpus_length=dimension))
    print('AVERAGE AMOUNT OF WORDS - simple', amount_words(2, final_corpu=final_corpus, corpus_length=dimension))

    which_stop1, average_amount1 = stopwords(1, final_corpu=final_corpus ,filter=False,  corpus_length= dimension, save=True)
    which_stop11, average_amount11 = stopwords(1, final_corpu=final_corpus, filter=True, corpus_length= dimension, save=True)
    which_stop_2, average_amount2 = stopwords(2, final_corpu=final_corpus, filter=False, corpus_length= dimension, save=True)
    which_stop22, average_amount22 = stopwords(2, final_corpu=final_corpus, filter=True,  corpus_length= dimension, save=True)
    print()
    print(average_amount1, average_amount11, average_amount2, average_amount22)

    plot_stop(stopwords=which_stop1, filter='not filtered', original='original', color='olive')
    plot_stop(stopwords=which_stop11, filter='filtered', original='original', color='darksalmon')
    plot_stop(stopwords=which_stop_2, filter='not filtered', original='simplified', color='lightblue')
    plot_stop(stopwords=which_stop22, filter='filtered', original='simplified', color='blue')

    key1, value1 = PoS_analysis(INTERMEDIATE_DIR + '/ADV_no_stop_final_1.txt')
    key11, value11 = PoS_analysis(INTERMEDIATE_DIR + '/no_stop_final_2.txt')
    key2, value2 = PoS_analysis(INTERMEDIATE_DIR + '/ADV_no_stop_final_2.txt')
    key22, value22 = PoS_analysis(INTERMEDIATE_DIR + '/no_stop_final_2.txt')
    print('Done!! ')

    barplot_PoS(key1, value1, value2)
    barplot_PoS(key11, value11, value22)

    mood_complex = verbal_features(typev='Mood',
                                   filename=INTERMEDIATE_DIR + '/ADV_no_stop_final_1.txt')
    mood_simple = verbal_features(typev='Mood',
                                  filename=INTERMEDIATE_DIR + '/ADV_no_stop_final_2.txt')
    tense_complex = verbal_features(typev='Tense',
                                    filename=INTERMEDIATE_DIR + '/ADV_no_stop_final_1.txt')
    tense_simple = verbal_features(typev='Tense',
                                   filename=INTERMEDIATE_DIR + '/ADV_no_stop_final_2.txt')
    print('Done')
    double_barplot(mood_complex, mood_simple, 'indianred', 'lightsalmon', 'Mood')
    double_barplot(tense_complex, tense_simple, 'dodgerblue', 'skyblue', 'Tense')

    # Calculating the amount of SVO structures in complex and simple sentences
    print(f'The amount of SVO in Original sentences is {amount_SVO(final_corpus, 1)}')
    print(f'The amount of SVO in Simplified sentences is {amount_SVO(final_corpus, 2)}')

    # Calculating the amount of coordination conjunctions are present in the complex and simple text
    tot_complex, collection_complex = coord_subord(final_corpus, 1, 'CCONJ')
    tot_simple, collection_simple = coord_subord(final_corpus, 2, 'CCONJ')
    tot_complex1, collection_complex1 = coord_subord(final_corpus, 1, 'SCONJ')
    tot_simple1, collection_simple1 = coord_subord(final_corpus, 2, 'SCONJ')
    print(tot_complex, tot_simple, tot_complex1, tot_simple1)

    print('The end!')