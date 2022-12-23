from collections import OrderedDict

import plotly.graph_objects as go

from settings import *
from src.analysis.utils_charts.stopwords_distribution import is_number

def PoS_analysis(filename):
    '''
    This function perform PoS tagging on the sentences given in input and calculate the frequency of each tag
    :param filename: you can provide the txt file containing complex or simple sentences, depending on your interests
    :return: a Tuple with the keys and the values of the frequency dictionary, sorted for key (n alphabetical order)
    '''


    sentences_final = []
    with open(filename, "r") as infile:
        for line in infile.readlines():
            line = line.strip()
            sentences_final.append(line)

    dict_pos = {}
    for sent in sentences_final:
        doc = nlp(sent)

        for token in doc:

            if not token.is_punct and not str(token).isdigit() and not is_number(str(token)):

                # excluding some part of speech from the analysis
                if token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and token.pos_ != 'PART' and token.pos_ != 'SYM' and token.pos_ != 'X' and token.pos_ != 'INTJ':

                    if token.pos_ not in dict_pos:

                        dict_pos[token.pos_] = 1
                    else:
                        dict_pos[token.pos_] += 1

    ordered_dict = OrderedDict(sorted(dict_pos.items(), key=lambda t: t[0]))

    return list(ordered_dict.keys()), list(ordered_dict.values())


def barplot_PoS(keys, v1, v2):
    '''
    This function allows to plot a grouped barplot which compares the amount of part of speech tags in the two types of phrases (complex and simplified)
    :param keys: sorted list of Part of Speech
    :param v1: Values of the first type of sentences (complex)
    :param v2: Values of the second type of sentences (simple)
    :return: a double barplot
    '''

    # calculate the frequency in percentage
    tot1 = sum(v1)
    tot2 = sum(v2)
    values1_new = [round((ele / tot1) * 100, 2) for ele in v1]
    values2_new = [round((ele / tot2) * 100, 2) for ele in v2]

    pos = list(keys)

    fig = go.Figure(data=[
        go.Bar(name='Complex', x=pos, y=values1_new),
        go.Bar(name='Simple', x=pos, y=values2_new)
    ])

    # Set the correct bar mode
    fig.update_layout(barmode='group', title_text='Comparison of distribution of PoS in Complex and Simple sentences',
                      title_x=0.5, font=dict(size=27))
    fig.update_xaxes(title_text="Part of Speech")
    fig.update_yaxes(title_text="Average percentage of PoS in the corpus")
    fig.show()
    return
