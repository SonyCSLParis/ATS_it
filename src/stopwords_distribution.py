import spacy
from spacy.lang.it.examples import sentences
import numpy as np
import pandas as pd
import csv
import plotly.express as px
from settings import *

def is_number(n):
    '''
    This function controls if the string we give in input is a number or not.
    :return: a boolean, True if the string is a number and False otherwise.
    '''
    is_number = True
    try:
        num = float(n)
        # check for "nan" floats
        is_number = num == num   # or use `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number


def amount_words(type_sent, final_corpu, corpus_length=47388):
    '''
    This function count the average word length of the sentences of PaCCS-IT corpus.
    :param type_sent: which type of sentences we want to analyze, could be either 1 (complex) or 2 (simplified)
    :param final_corpu: is the corpus we are referring to
    :param corpus_length: is the length of the corpus (amount of rows)
    :return: a tuple containing the total amount of words in the corpus and the average sentence length
    '''
    amount_of_words = 0
    for sentence in final_corpu[f'Sentence_{type_sent}']:
        length = len(sentence.split())
        amount_of_words += length
    if type_sent == 1:
        t = 'Complex'
    else:
        t = 'Simple'

    return amount_of_words, f'The average amount of words for {t} Sentences is {round(amount_of_words / corpus_length, 2)}'


def stopwords(type_sent, final_corpu, filter=None, corpus_length=47388, save = False):
    '''
    This function calculate the amount of stopwords present in the stopwords.
    :param type_sent: which type of sentences we want to analyze, could be either 1 (complex) or 2 (simplified)
    :param final_corpu: is the corpus we are referring to
    :param filter: if True it allows to eliminate some stopwords from the original list
    :param corpus_length:
    :param save: if True it allows to save the new sentences, without stopwords into a txt file
    :return: a Tuple containing as first argument the list of stopwords that were found and the average amount of stopwords for sentences
    '''

    if filter:
        #nlp.Defaults.stop_words |= {"i","I", 'molte', 'molti'}
        stopword = nlp.Defaults.stop_words
        list_stop = list(stopword)
        set_stop = ' '.join(list_stop)
        doc = nlp(set_stop)
        for token in doc:
            word, pos = (token.text, token.pos_)
            #we want to keep in the sentences all the verbs, auxiliar and adverbs that were considered to be stopwords
            #we need to understand if it's a good idea to keep or exclude the ADV
            if pos == 'VERB' or pos == 'AUX' or pos == 'ADV':
                if pos != '"' and pos != 's':

                    stopword.remove(word)

    else:
        stopword = nlp.Defaults.stop_words

    count = 0
    which_stopwords = []
    sentences_final = []
    for sentence in final_corpu[f'Sentence_{type_sent}']:
        lst = []
        for token in sentence.split():
            if token.lower() not in stopword and not token.isdigit() and not is_number(token):
                lst.append(token)

            else:
                which_stopwords.append(token.lower())
                count += 1

        sentences_final.append(' '.join(lst))

    if type_sent == 1:
        t = 'Complex'
    else:
        t = 'Simple'

    if save:
        with open(OUTPUT_DIR + '/ADV_no_stop_sentences_{type_sent}.txt', 'w') as fp:
            for item in sentences_final:
                # write each item on a new line
                fp.write("%s\n" % item)


    return which_stopwords, f'Average amount of stopwords for {t} Sentences is {round(count / corpus_length, 2)}'




def plot_stop(stopwords, filter, original, color):
    '''
    This function allows to produce a bar plot of the most frequent stopwords in the corpus' sentences
    :param stopwords: the list of stopwords that you can obtain from the stopwords function
    :param filter: 'filtered' if you are dealing with filtered stopwords or 'not filtered' if you keep the original set of stopwords
    :param original: 'original' or 'simplified', depends on which sentences you are analyzing
    :param color: you can decide the color of the bars
    :return: an interactive barplot produced with plotly
    '''
    #Create a dictionary with stopwords frequency in the corpus and then choose the 25 most frequent
    dict_stop = {}
    for ele in stopwords:
        if ele in dict_stop:
            dict_stop[ele] +=1
        else:
            dict_stop[ele] = 1

    sorted_dict = dict(reversed(sorted(dict_stop.items(), key = lambda x:x[1])))

    i = 0
    keys = []
    values = []

    for k,v in sorted_dict.items():
        if i < 25:
            keys.append(k)
            values.append(v)
        i +=1

    data_tuples = list(zip(keys,values))
    df = pd.DataFrame(data_tuples, columns=['Stopwords','Frequency'])

    '''fig = px.bar(df, x="Frequency", y="Stopwords", orientation='h', title=f'Frequency of {filter} stopwords in {original} sentences')
    fig.update_traces(marker_color=color)
    fig.show()'''


    fig2 = px.bar(df, x='Stopwords', y='Frequency', orientation='v', title= f'Frequency of {filter} stopwords in {original} sentences')
    fig2.update_traces(marker_color=color)
    fig2.update_layout(font = dict(size = 18))
    fig2.show()


    return




