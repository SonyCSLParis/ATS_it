from settings import *
import csv
import pandas as pd
pacssit = INCOMPLETE_DATASET_DIR + '/pacs_pulito_finale.csv'
teacher = INCOMPLETE_DATASET_DIR + '/teacher_1.csv'
terence = INCOMPLETE_DATASET_DIR + '/terence_1.csv'
simptiki = INCOMPLETE_DATASET_DIR + '/simpitiki_1.csv'
all_together = HF_DATASETS + '/finilized_dataset_1.csv'



df = pd.read_csv(simptiki)

df = df[['Normal', 'Simple']]
df1 = pd.read_csv(teacher)
df1 = df1[['Normal', 'Simple']]
df2 = pd.read_csv(terence)
df2 = df2[['Normal', 'Simple']]
df3 = pd.read_csv(pacssit)

df3 = df3[['Normal', 'Simple']]

df4 = pd.concat([df, df1, df2, df3])
df4.to_csv(all_together, index=False)