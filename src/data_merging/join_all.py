from settings import *
import csv
import pandas as pd
pacssit = INCOMPLETE_DATASET_DIR + '/final_hf.csv'
teacher = INTERMEDIATE_DIR + '/incomplete_datasets/teacher.csv'
terence = INTERMEDIATE_DIR + '/incomplete_datasets/terence.csv'
simptiki = INTERMEDIATE_DIR + '/incomplete_datasets/simpitiki.csv'
all_together = OUTPUT_DIR + '/output_modello/tts.csv'



df = pd.read_csv(simptiki)

df = df[['Sentence_1', 'Sentence_2']]
df1 = pd.read_csv(teacher)
df1 = df1[['Sentence_1', 'Sentence_2']]
df2 = pd.read_csv(terence)
df2 = df2[['Sentence_1', 'Sentence_2']]
df3 = pd.read_csv(pacssit)

df3 = df3[['Sentence_1', 'Sentence_2']]

df4 = pd.concat([df, df1, df2])
df4.to_csv(all_together, index=False)