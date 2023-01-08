import csv
import pandas as pd
from settings import *
nlp = spacy.load('it_core_news_sm')
avg_cos_sim = 0.782

'''
the functions clean_sentence_1, lemmatize_text and clean_corpus are specifically designed for paccsit only, while the last function
called eliminate_punct can be used for the other three dataset
'''

def clean_sentence_1(sentence):
    # create the doc
    doc1 = nlp(sentence)

    # filter out stop_words, punctuation and non-latin characters
    filtered_s = [token for token in doc1 if token.is_punct == False and token.is_ascii == True and token.is_stop == False]

    # Rebuild a list of strings
    filtered_ss = [str(token) for token in filtered_s]

    # if the string is longer than 0, recreate the string
    if len(filtered_ss) > 0:
        final = ' '.join(filtered_ss)

    # otherwise assign None to it
    else:
        final = None

    return final


def lemmatize_text(doc):

    #you get the lemma only of verbs and auxiliaries, the rest of teh words are kept as the original
    tokens = [str(token.lemma_) if token.pos_ == 'VERB' or token.pos_ == 'AUX' else str(token.text) for token in doc ]

    output = ' '.join(tokens)

    return output



def clean_corpus(operation, input_file, output_file):
    list_complex = []
    list_simple = []
    list_of_cos = []


    #open the complete dataset as a csv
    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        #skip header
        next(csv_reader)

        #read each row
        for row in csv_reader:

            #read each sentence of the raw
            for i in range(len(row)):

                #if we are dealing with the complex sentences
                if i == 0:
                    complex_s = clean_sentence_1(row[i])
                    doc1 = nlp(row[0])

                #if we are dealing with the simple sentences
                if i == 1:
                    simplified_s = clean_sentence_1(row[i])
                    doc2 = nlp(row[1])

            #if both the sentences are different from None, I calculate their cosine similarity
            if complex_s != None and simplified_s != None:
                d1 = nlp(complex_s)
                d2 = nlp(simplified_s)
                similarity = d1.similarity(d2)
                list_of_cos.append(similarity)


            if similarity > 0.05 and similarity < 0.95:


                if operation == 'lemmatization':

                    #just lemmatize the complex and simple words
                    lem_complex = lemmatize_text(doc1)
                    list_complex.append(lem_complex)

                    lem_simple = lemmatize_text(doc2)
                    list_simple.append(lem_simple)



                elif operation == 'lemmatiz_stopwords':

                    len_frase_1 = len(doc1)
                    stopwords_1 = [token for token in doc1 if token.is_stop]

                    len_frase_2 = len(doc2)
                    stopwords_2 = [token for token in doc2 if token.is_stop]

                    #if neither the first or the second sentence are made up of all stopwords, then lemmatize them
                    if len_frase_1 != len(stopwords_1) and len_frase_2 != len(stopwords_2):

                        lem_complex = lemmatize_text(doc1)
                        list_complex.append(lem_complex)

                        lem_simple = lemmatize_text(doc2)
                        list_simple.append(lem_simple)


                elif operation == 'general_filtering':

                    #this time I filter out only for puntuation
                    to_charge = [token for token in doc1 if token.is_punct == False ]

                    # I rebuild the list of strings
                    to_charge1 = [str(token).lower() for token in to_charge]

                    #if the list of string is longer than 2
                    if len(to_charge1) > 2:

                        # I recreate the string
                        f = ' '.join(to_charge1)

                        if f not in list_complex:

                            # append it to the list of complex sentences
                            list_complex.append(f)

                            #do the same thing for the simplified sentence
                            to_charge_sim = [token for token in doc2 if token.is_punct == False ]
                            to_charge1_sim = [str(token).lower() for token in to_charge_sim]

                            f_sim = ' '.join(to_charge1_sim)
                            list_simple.append(f_sim)

    # I create the final dataset and I save it as a csv
    d = {'Normal': list_complex, 'Simple': list_simple}
    df = pd.DataFrame(d)
    df.to_csv(output_file, index=False)

    return

clean_corpus('general_filtering')

'''clean_corpus('general_filtering', input_file='/Users/francesca/Desktop/documenti_potenziali/data.v1.split/dataset_tradotto_second.csv', output_file='/Users/francesca/Desktop/dataset_utilizzati/dataset_tradotto_pulito.csv' )
clean_corpus('general_filtering', input_file='/Users/francesca/Desktop/dataset_utilizzati/training_italiano.csv', output_file='/Users/francesca/Desktop/dataset_utilizzati/training_italiano_pulito.csv')
clean_corpus('general_filtering', input_file='/Users/francesca/Desktop/dataset_utilizzati/development_italiano.csv', output_file= '/Users/francesca/Desktop/dataset_utilizzati/development_italiano_pulito.csv')
'''
def eliminate_punct(input_file, output_file):
    list_complex = []
    list_simple = []

    # open the complete dataset as a csv
    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # skip header
        next(csv_reader)

        # read each row
        for row in csv_reader:

            # read each sentence of the raw
            for i in range(len(row)):

                if i == 1:

                    prima = nlp(row[1])
                    prima1 = [str(token).lower() for token in prima if not token.is_punct]
                    prima1_1 = ' '.join(prima1)


                elif i == 2:
                    seconda = nlp(row[2])
                    seconda1 = [str(token).lower() for token in seconda if not token.is_punct]
                    seconda1_1 = ' '.join(seconda1)


            list_complex.append(prima1_1)
            list_simple.append(seconda1_1)


    d = {'Normal': list_complex, 'Simple': list_simple}
    df = pd.DataFrame(d)
    df.to_csv(output_file, index=False)

    return


'''eliminate_punct(input_file= INCOMPLETE_DATASET_DIR +'/simpitiki_1.csv', output_file= '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/simpitiki_2.csv')
eliminate_punct(input_file= INCOMPLETE_DATASET_DIR + '/teacher.csv', output_file= INCOMPLETE_DATASET_DIR + '/teacher_1.csv')
eliminate_punct(input_file= INCOMPLETE_DATASET_DIR + '/terence.csv', output_file= INCOMPLETE_DATASET_DIR + '/terence_1.csv')'''


























