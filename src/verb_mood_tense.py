import spacy
from collections import OrderedDict
import plotly.graph_objects as go
from stopwords_distribution import is_number
from subject_verb_object_extract import findSVOs


def verbal_features(typev, filename):
    '''
    This function manage to calculate the frequency of morphological verbal features in the text.
    :param typev: can be either 'Mood' or 'Tense' with respect to what you are interested in
    :param filename: is the txt file referring to the simple or complex sentences
    :return: the dictionary containing verbal features frequency
    '''

    nlp = spacy.load("it_core_news_sm")
    stopwords = nlp.Defaults.stop_words

    sentences_final = []
    with open(filename, "r") as infile:
        for line in infile.readlines():
            line = line.strip()
            sentences_final.append(line)

    collection = {}
    for sent in sentences_final:
        doc = nlp(sent)
        for token in doc:
            if token.pos_ == 'VERB':
                if token.morph.get(typev):
                    feature = token.morph.get(typev)[0]
                    if feature not in collection:
                        collection[feature] = 1
                    else:
                        collection[feature] += 1

    return collection




def double_barplot(complex, simple, color1, color2):
    '''
    This function allows to produce a double barplot which compares the distribution of verbal features in complex and simple sentences
    :param complex: dictionary of verbal features frequency obtained by analyzing complex sentences
    :param simple: dictionary of verbal features frequency obtained by analyzing simple sentences
    :return: plot
    '''
    ordered_dict1 = OrderedDict(sorted(simple.items(), key=lambda t: t[0]))
    ordered_dict2 = OrderedDict(sorted(complex.items(), key=lambda t: t[0]))

    key1,value1 = list(ordered_dict1.keys()), list(ordered_dict1.values())
    key2, value2 = list(ordered_dict2.keys()), list(ordered_dict2.values())


    tot_simple_1 = sum(value1)
    y_simple_final = [round(ele/tot_simple_1, 2) for ele in value1]

    tot_complex_1 = sum(value2)
    y_complex_final = [round(ele/tot_complex_1, 2) for ele in value2]


    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=key1,
        y=y_simple_final,  # aggiungo +200 agli ultimi due
        name='Simple Sentences',
        marker_color=color1
    ))
    fig.add_trace(go.Bar(
        x=key1,
        y=y_complex_final,  # aggiungo +200 agli ultimi due
        name='Complex Sentences',
        marker_color=color2
    ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title_text='Mood distribution in Complex and Simple Sentences', font = dict(size = 29))
    fig.update_traces(width=0.4)
    fig.update_xaxes(title_text="Type of verbal feature")
    fig.update_yaxes(title_text="Average percentage of moods in the corpus")
    fig.show()
    return


def coord_subord(corpus, sentence_type, which_POS):
    '''
    This function allows to calculate the amount of CCONJ and SCONJ are present in the corpus.
    :param corpus: The corpus we are referring to
    :param sentence_type: int 1 (complex) or 2 (simple
    :param which_POS: could be either 'CCONJ' or 'SCONJ'
    :return: tuple with the total amount of the particel and the dictionary of their frequency
    '''
    nlp = spacy.load("it_core_news_sm")
    coll = {}
    for sentence in corpus[f'Sentence_{sentence_type}']:
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ == which_POS:
                if not is_number(token.text):

                    if token.text not in coll:
                        coll[token.text] = 1

                    coll[token.text] += 1

    return sum(list(coll.values())), coll


def amount_SVO(corpus,sentence_type):
    '''
    This function allows to calculate the amount of Subject Verb Object phrases present in the corpus sentences
    :param corpus: corpus we are referring to
    :param sentence_type: 1 (complex) or 2 (simple)
    :return: amount of SVO
    '''
    count = 0
    nlp = spacy.load("it_core_news_sm")
    for sentence in corpus[f'Sentence_{sentence_type}']:
        doc = nlp(sentence)
        svos = findSVOs(doc)
        if len(svos) > 0:
            count +=1

    return count