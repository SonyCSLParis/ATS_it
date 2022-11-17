from src.data_merging.settings import *
import csv
import pandas as pd
pacssit = INCOMPLETE_DATASET_DIR + 'final_hf.csv'
teacher = INTERMEDIATE_DIR + 'teacher.csv'
terence = INTERMEDIATE_DIR + 'terence.csv'
simptiki = INTERMEDIATE_DIR + 'simpitiki.csv'
all_together = OUTPUT_DIR + 'ultimated.csv'
'''with open(all_together, 'a') as f_object:

    writer_object = csv.writer(f_object)
    header = ['index','Sentente_1', 'Sentence_2']

    with open(pacssit, 'r') as infile0:
        next(infile0) #skip header
        for riga in infile0:
            print(riga[0], riga[1], riga[2])
            writer_object.writerow(riga.split(','))

    with open(teacher, 'r') as infile:
        next(infile)  # skip header
        for riga in infile:
            writer_object.writerow(riga.split(','))

    with open(terence, 'r') as infile2:
        next(infile2)  # skip header
        for riga in infile2:

            writer_object.writerow()

    with open(simptiki, 'r') as infile3:
        next(infile3)  # skip header
        for riga in infile3:
            writer_object.writerow(riga.split(','))

'''



df = pd.read_csv(pacssit)
df = df[['Sentence_1', 'Sentence_2']]
df1 = pd.read_csv(teacher)
df1 = df1[['Sentence_1', 'Sentence_2']]
df2 = pd.read_csv(terence)
df2 = df2[['Sentence_1', 'Sentence_2']]
#df3 = pd.read_csv(simptiki)

df4 = pd.concat([df, df1, df2])
df4.to_csv(all_together, index=False)