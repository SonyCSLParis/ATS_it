import pandas as pd
import csv
from settings import *
import os
'''
def delete_tokens(dataset):

    colonna_complessa_train = [str(riga) for riga in list(dataset['Normal'])]
    cleaned = []

    for compl in colonna_complessa_train:
        sentence = compl.split()
        sentence_2 = sentence[6:]
        sentence_3 = ' '.join(sentence_2)
        cleaned.append(sentence_3)

    return cleaned



def filter_test(path):

    for folder in os.listdir(path):

        if folder != '.DS_Store' and folder != 'adaptive':
            joined = CSV_FILES_PATH + '/' + folder

            for ele in ['/train.csv', '/test.csv', '/val.csv']:

                joined_1 = joined + ele

                if 'train' in joined_1:
                    df_train = pd.read_csv(joined_1)
                    colonna_complessa_train = [str(riga) for riga in list(df_train['Normal'])]

                elif 'test' in joined_1:
                    df_test = pd.read_csv(joined_1)
                    colonna_complessa_test = [str(riga) for riga in list(df_test['Normal'])]
                    colonna_simple_test = [str(riga) for riga in list(df_test['Simple'])]

                elif 'val' in joined_1:
                    df_val = pd.read_csv(joined_1)
                    colonna_complessa_val = [str(riga) for riga in list(df_val['Normal'])]

                lista_test_complex = []
                lista_test_simple = []

            for i in range(len(colonna_complessa_test)):
                if colonna_complessa_test[i] not in colonna_complessa_train and colonna_complessa_test[i] not in colonna_complessa_val:
                    lista_test_complex.append(colonna_complessa_test[i])
                    lista_test_simple.append(colonna_simple_test[i])

            with open(joined + '/test_filtered.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(lista_test_complex, lista_test_simple))



        elif folder != '.DS_Store' and folder == 'adaptive':

            joined = CSV_FILES_PATH + '/' + folder
            df_train = pd.read_csv(joined + '/train.csv')
            df_test = pd.read_csv(joined + '/test.csv')
            df_val = pd.read_csv(joined + '/val.csv')

            colonna_complessa_train = [str(riga) for riga in list(df_train['Normal'])]
            cc_trained_no_tok = delete_tokens(df_train)

            colonna_complessa_test = [str(riga) for riga in list(df_test['Normal'])]
            cc_test_no_tok = delete_tokens(df_test)

            colonna_simple_test = [str(riga) for riga in list(df_test['Simple'])]
            cs_test_no_tok = delete_tokens(df_test)

            colonna_complessa_val = [str(riga) for riga in list(df_val['Normal'])]
            cc_val_no_tok = delete_tokens(df_val)

            lista_test_complex = []
            lista_test_simple = []

            for i in range(len(cc_test_no_tok)):
                if cc_test_no_tok[i] not in cc_trained_no_tok and cc_test_no_tok[i] not in cc_val_no_tok:
                    lista_test_complex.append(colonna_complessa_test[i])
                    lista_test_simple.append(colonna_simple_test[i])

            with open(joined + '/test_filtered.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(lista_test_complex, lista_test_simple))

    return


filter_test(CSV_FILES_PATH)'''

joined = CSV_FILES_PATH + '/' + 'adap_enriched'

for ele in ['/train.csv', '/test.csv', '/val.csv']:

    joined_1 = joined + ele

    if 'train' in joined_1:
        df_train = pd.read_csv(joined_1)
        colonna_complessa_train = [str(riga) for riga in list(df_train['Normal'])]

    elif 'test' in joined_1:
        df_test = pd.read_csv(joined_1)
        colonna_complessa_test = [str(riga) for riga in list(df_test['Normal'])]
        colonna_simple_test = [str(riga) for riga in list(df_test['Simple'])]

    elif 'val' in joined_1:
        df_val = pd.read_csv(joined_1)
        colonna_complessa_val = [str(riga) for riga in list(df_val['Normal'])]

    lista_test_complex = []
    lista_test_simple = []

for i in range(len(colonna_complessa_test)):
    if colonna_complessa_test[i] not in colonna_complessa_train and colonna_complessa_test[i] not in colonna_complessa_val:
        lista_test_complex.append(colonna_complessa_test[i])
        lista_test_simple.append(colonna_simple_test[i])

with open(joined + '/test_filtered.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(lista_test_complex, lista_test_simple))