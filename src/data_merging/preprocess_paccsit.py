import spacy
import csv
import pandas as pd

nlp = spacy.load('it_core_news_sm')

data_input = '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/pacs_number.csv'
data_output = '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/pacs_pulito_finale.csv'
file_trat = '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/pronomi.csv'

def clean_corpus(input_file, output_file, file_trattini = None):
    list_complex = []
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
                    '''check = row[1].split()
                    check2 = row[2].split()
                    for i in range(len(check) -1):
                        if '-' in check[i] and len(check[i+1]) == 2 and check[i+1].isalpha():
                            lista_complessa_tr.append(row[1])
                            lista_semplice_tr.append(row[2])

                    for i in range(len(check2) - 1):
                        if '-' in check2[i] and len(check2[i + 1]) == 2 and check2[i + 1].isalpha():
                            lista_complessa_tr.append(row[1])
                            lista_semplice_tr.append(row[2])'''

                    prima = nlp(row[0])
                    prima1 = [str(token).lower() for token in prima if not token.is_punct]
                    prima1_1 = ' '.join(prima1)



                elif i == 1:
                    seconda = nlp(row[1])
                    seconda1 = [str(token).lower() for token in seconda if not token.is_punct]
                    seconda1_1 = ' '.join(seconda1)


            if prima1_1 not in list_complex:
                list_complex.append(prima1_1)
                list_simple.append(seconda1_1)

            elif prima1_1 in list_complex:
                if seconda1_1 not in list_simple:
                    list_complex.append(prima1_1)
                    list_simple.append(seconda1_1)


    d = {'Normal': list_complex, 'Simple': list_simple}
    df = pd.DataFrame(d)
    df.to_csv(output_file, index=False)

    '''d2 = {'Normal': lista_complessa_tr, 'Simple': lista_semplice_tr}
    df2 = pd.DataFrame(d2)
    df2.to_csv(file_trattini, index=False)'''

    return


clean_corpus(input_file=data_input, output_file= data_output, file_trattini = None)


to_be_adjusted = '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/pacs_pulito.csv'
adjusted = '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/pacs_number.csv'

def adjust_number(input_file, output_file):

    lista_c = []
    lista_s = []

    # open the complete dataset as a csv
    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # skip header
        next(csv_reader)

        # read each row
        for row in csv_reader:

            complessa = row[0].split()
            semplice = row[1].split()

            lista_numeri_complessa = []

            for j in range(len(complessa)):

                if complessa[j].isdigit():
                    lista_numeri_complessa.append(complessa[j])

            numeri_complessi = list(reversed(lista_numeri_complessa))



            if numeri_complessi != []:

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


eliminate_punct(input_file= '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/simpitiki.csv', output_file= '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/simpitiki_1.csv')
eliminate_punct(input_file= '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/teacher.csv', output_file= '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/teacher_1.csv')
eliminate_punct(input_file= '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/terence.csv', output_file= '/Users/francesca/Desktop/Github/Final/intermediate/incomplete_datasets/terence_1.csv')












