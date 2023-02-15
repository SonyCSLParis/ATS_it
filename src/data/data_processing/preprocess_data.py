import csv
import os
from settings import *
import re

'''
The functions in this script are intended to parse the various corpus in use and apply a whole series of corrections and filters to our dataset.
'''


file_pronouns = INCOMPLETE_DATASET_DIR +'/pronomi.csv'

def adjust_number(sentence1, sentence2):
    '''
    This function allows you to correct errors within number-related sentences.
    Most of the sentences had causal numbers that were not homologous between the original and simple sentences.
    :param sentence1: complex sentence in input
    :param sentence2: simple sentence in input
    :return: complex and simple - parallel sentence corrected
    '''
    cc = sentence1.split()
    ss = sentence2.split()

    lista_numeri_complessa = []

    # I save the numbers I encounter in the complex sentence
    for j in range(len(cc)):

        if cc[j].isdigit():
            lista_numeri_complessa.append(cc[j])

    numeri_complessi = list(reversed(lista_numeri_complessa))

    if numeri_complessi != []:

        # I correct the numbers in the simplified sentence
        for j in range(len(ss)):

            if ss[j].isdigit():
                if len(numeri_complessi) != 0:
                    ele = numeri_complessi.pop()
                    ss[j] = ele

    frase_semplice = ' '.join(ss)

    return sentence1.strip(), frase_semplice.strip()


def clean_corpus(input_file, output_file, file_pronomi = None):
    '''
    This function filters the dataset according to some previously defined criteria.
    If you provide the variable file_pronomi, the function allows you to calculate
    how many sentences within the dataset have a - marking the presence of a pronoun in the sentence.
    Do it only with PaCCSS-IT.
    :param input_file: directory of the original corpus
    :param output_file: directory of the parsed corpus
    :param file_pronomi: directory of the file containing sentnces with marked pronouns
    '''

    list_complex = []
    list_simple = []

    lista_semplice_tr = []
    lista_complessa_tr = []


    #open the complete dataset as a csv
    with open(input_file, 'r') as in_file:
        csv_reader = csv.reader(in_file, delimiter=',')

        #skip header
        next(csv_reader)

        with open(output_file, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(['Normal', 'Simple'])
            #read each row
            for row in csv_reader:

                if file_pronomi:

                    for i in range(len(row)):

                        if i == 0:
                            check = row[0].split()
                            check2 = row[1].split()

                            #this part is needed to detect which sentences marks the pronoun on some verbs
                            for i in range(len(check) -1):
                                if '-' in check[i] and len(check[i+1]) == 2 and check[i+1].isalpha():
                                    lista_complessa_tr.append(row[0])
                                    lista_semplice_tr.append(row[1])

                            for i in range(len(check2) - 1):
                                if '-' in check2[i] and len(check2[i + 1]) == 2 and check2[i + 1].isalpha():
                                    lista_complessa_tr.append(row[0])
                                    lista_semplice_tr.append(row[1])


                #adjust the numbers in the sentences and eliminate punctuation at the beginning and at the end of the sentences
                ccomm, ssempp = adjust_number(row[1], row[2])
                res = re.sub(r'(^[^\w]+)|([^\w]+$)', '', ccomm)
                res1 = re.sub(r'(^[^\w]+)|([^\w]+$)', '', ssempp)

                #apply another strip control
                res = res.strip('\"')
                res1 = res1.strip('\"')

                #calculate cosine similarity
                doc1 = nlp(ccomm)
                doc2 = nlp(ssempp)
                similarity = doc1.similarity(doc2)

                #keep only instances with a similarity bigger than 0.05
                if similarity > 0.05:

                    #if the complex sentence is not in the list already, add it, together with the corresponding simple one
                    if res not in list_complex and not res1.startswith('articolo'):
                        writer.writerow([res, res1])
                        list_complex.append(res)
                        list_simple.append(res1)

                    elif res in list_complex and res1 not in list_simple or res in list_complex and res1 not in list_simple and res == res1:
                        writer.writerow([res, res1])
                        list_complex.append(res)
                        list_simple.append(res1)


    return


#process all the files in use, which are contained in various different directories
grounds = [TURKUS_TRANSLATED, INCOMPLETE_NO_PROCESSED, WIKIPEDIA_TRANSLATED]
for ground in grounds:
    if ground == TURKUS_TRANSLATED:
        for file in os.listdir(ground):
            if file != '.DS_Store':
                full = ground + '/' + file
                full_out = full[:-4] +'_processato.csv'
                clean_corpus(input_file= full, output_file= full_out)

    elif ground == INCOMPLETE_NO_PROCESSED:
        for file in os.listdir(ground):
            full = ground + '/' + file
            full_out = INCOMPLETE_PROCESSED + '/' + file[:-4] + '_processato.csv'
            print(full_out)
            clean_corpus(input_file=full, output_file=full_out)

    elif ground == WIKIPEDIA_TRANSLATED:
        for file in os.listdir(ground):
            if 'deepl' in file:
                full = ground + '/' + file
                full_out = full[:-4] + '_processato.csv'
                print(full_out)
                clean_corpus(input_file=full, output_file=full_out)


