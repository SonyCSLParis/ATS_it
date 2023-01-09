import csv
import pandas as pd
from src.settings import *

data_input = INCOMPLETE_DATASET_DIR + '/pacs_number.csv'
data_output = INCOMPLETE_DATASET_DIR + '/pacs_clean_final.csv'
file_pronouns = INCOMPLETE_DATASET_DIR +'/pronomi.csv'

def clean_corpus(input_file, output_file, file_pronomi = None):
    list_complex = []
    list_complex_fourtok = []
    list_simple = []

    lista_semplice_tr = []
    lista_complessa_tr = []


    #open the complete dataset as a csv
    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        #skip header
        next(csv_reader)

        #read each row
        for row in csv_reader:

            #read each sentence of the raw
            for i in range(len(row)):

                if i == 0:
                    check = row[0].split()
                    check2 = row[1].split()

                    if file_pronomi:
                        #this part is needed to detect which sentences marks the pronoun on some verbs
                        for i in range(len(check) -1):
                            if '-' in check[i] and len(check[i+1]) == 2 and check[i+1].isalpha():
                                lista_complessa_tr.append(row[0])
                                lista_semplice_tr.append(row[1])

                        for i in range(len(check2) - 1):
                            if '-' in check2[i] and len(check2[i + 1]) == 2 and check2[i + 1].isalpha():
                                lista_complessa_tr.append(row[0])
                                lista_semplice_tr.append(row[1])

            prima = nlp(row[0])
            prima1 = [str(token).lower() for token in prima if not token.is_punct]
            prima1_1 = ' '.join(prima1)


            seconda = nlp(row[1])
            seconda1 = [str(token).lower() for token in seconda if not token.is_punct]
            seconda1_1 = ' '.join(seconda1)


            #avoid to add the sentences which are completely equal
            if prima1_1 not in list_complex:

                #avoid to add the sentences which are very similar, that they have the first 6 tokens in common and also the complex sentences that starts with 'articolo'
                if ' '.join(prima1_1.split()[:6]) not in list_complex_fourtok and prima1_1.split()[0] != 'articolo':

                        doc1 = nlp(prima1_1)
                        doc2 = nlp(seconda1_1)
                        similarity = doc1.similarity(doc2)

                        if similarity > 0.05 and similarity < 0.95:


                            list_complex.append(prima1_1)
                            list_complex_fourtok.append(' '.join(prima1_1.split()[:4]))
                            list_simple.append(seconda1_1)


    d = {'Normal': list_complex, 'Simple': list_simple}
    df = pd.DataFrame(d)
    df.to_csv(output_file, index=False)

    d2 = {'Normal': lista_complessa_tr, 'Simple': lista_semplice_tr}
    df2 = pd.DataFrame(d2)
    df2.to_csv(file_pronomi, index=False)

    return


#clean_corpus(input_file='/Users/francesca/Desktop/Github/Final_final/output/csv_files/augmented/augmented_dataset.csv', output_file= '/Users/francesca/Desktop/Github/Final_final/output/csv_files/augmented/augmented_dataset_1.csv', file_pronomi = None)
clean_corpus(input_file='/Users/francesca/Desktop/Github/Final_final/output/csv_files/paccssit/paccss_only.csv', output_file= '/Users/francesca/Desktop/Github/Final_final/output/csv_files/paccssit/paccss_only_1.csv', file_pronomi = None)
#clean_corpus(input_file='/Users/francesca/Desktop/Github/Final_final/output/csv_files/finalized_dataset/finalized_df.csv', output_file= '/Users/francesca/Desktop/Github/Final_final/output/csv_files/finalized_dataset/finalized_df_1.csv', file_pronomi = None)


to_be_processed = INCOMPLETE_DATASET_DIR + '/pacs_clean.csv'
processed = INCOMPLETE_DATASET_DIR + '/pacs_number.csv'

def adjust_number(input_file, output_file):

    lista_c = []
    lista_s = []

    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        next(csv_reader)

        for row in csv_reader:

            complessa = row[0].split()
            semplice = row[1].split()

            lista_numeri_complessa = []

            #I save the numbers I encounter in the complex sentence
            for j in range(len(complessa)):

                if complessa[j].isdigit():
                    lista_numeri_complessa.append(complessa[j])

            numeri_complessi = list(reversed(lista_numeri_complessa))


            if numeri_complessi != []:

                #I correct the numbers in the simplified sentence
                for j in range(len(semplice)):

                    if semplice[j].isdigit():
                        if len(numeri_complessi) != 0:
                            ele = numeri_complessi.pop()
                            semplice[j] = ele


            frase_semplice = ' '.join(semplice)

            lista_c.append(row[0])
            lista_s.append(frase_semplice)

    d = {'Normal': lista_c, 'Simple': lista_s}
    df = pd.DataFrame(d)
    df.to_csv(output_file, index=False)

    return


#adjust_number(input_file=to_be_adjusted, output_file=adjusted)





