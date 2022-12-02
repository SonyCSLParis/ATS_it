import spacy
import csv
import pandas as pd

nlp = spacy.load('it_core_news_sm')


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

    tokens = [str(token.lemma_) if token.pos_ == 'VERB' or token.pos_ == 'AUX' else str(token.text) for token in doc ]

    output = ' '.join(tokens)

    return output

def lemmatize_text_nostop(doc):

    tokens = [str(token.lemma_) if token.pos_ == 'VERB' or token.pos_ == 'AUX' else str(token.text) for token in doc ]

    output = ' '.join(tokens)

    return output



def clean_corpus(operation, input_file, output_file):
    list_complex = []
    list_simple = []
    list_of_cos = []

    average_sim = 0.782

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
                list_of_cos.append(d1.similarity(d2))

            #I set the value to 0.7 because the average cosine similarity is 0.78
            if similarity > 0.6:


                if operation == 'lemmatization':

                    lem_complex = lemmatize_text(doc1)
                    list_complex.append(lem_complex)

                    lem_simple = lemmatize_text(doc2)
                    list_simple.append(lem_simple)



                elif operation == 'lemmatiz_stopwords':

                    len_frase_1 = len(doc1)
                    stopwords_1 = [token for token in doc1 if token.is_stop]

                    len_frase_2 = len(doc2)
                    stopwords_2 = [token for token in doc2 if token.is_stop]

                    if len_frase_1 != len(stopwords_1) and len_frase_2 != len(stopwords_2):

                        lem_complex = lemmatize_text(doc1)
                        list_complex.append(lem_complex)

                        lem_simple = lemmatize_text(doc2)
                        list_simple.append(lem_simple)


                elif operation == 'general_filtering':


                    #this time I filter out only for puntuation and non-latin characters
                    to_charge = [token for token in doc1 if token.is_punct == False and token.is_ascii == True]

                    # I rebuild the list of strings
                    to_charge1 = [str(token) for token in to_charge]

                    #if the list of string is longer than 2
                    if len(to_charge1) > 2:

                        # I recreate the string
                        f = ' '.join(to_charge1)

                        # append it to the list of complex sentences
                        list_complex.append(f)

                        #do the same thing for the simplified sentence
                        to_charge_sim = [token for token in doc2 if token.is_punct == False and token.is_ascii == True]
                        to_charge1_sim = [str(token) for token in to_charge_sim]

                        f_sim = ' '.join(to_charge1_sim)
                        list_simple.append(f_sim)

    # I create the final dataset and I save ita s a csv
    d = {'Normal': list_complex, 'Simple': list_simple}
    df = pd.DataFrame(d)
    df.to_csv(output_file, index=False)

    return



data_input = '/Users/francesca/Desktop/Github/Final/output/output_modello/ultimated.csv'
data_output = '/Users/francesca/Desktop/Github/Final/output/output_modello/lem_nostop_ultimated.csv'

clean_corpus(operation='lemmatiz_stopwords', input_file=data_input, output_file=data_output)


















