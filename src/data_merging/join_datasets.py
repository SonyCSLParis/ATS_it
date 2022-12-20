from settings import *
import pandas as pd

#this is the pacssit dataset without double sentences and correct with respect to the numbers, everything in lower case
pacssit = INCOMPLETE_DATASET_DIR + '/pacs_pulito_finale.csv'

#teacher, terence and simpitiki in lower case
teacher = INCOMPLETE_DATASET_DIR + '/teacher_1.csv'
terence = INCOMPLETE_DATASET_DIR + '/terence_1.csv'
simpitiki = INCOMPLETE_DATASET_DIR + '/simpitiki_2.csv'

#path of the joined dataset
all_together = HF_DATASETS + '/finilized_dataset_1.csv'


#open the dataset and select the columns of interest, eventually join everything
df = pd.read_csv(pacssit)
df = df[['Normal', 'Simple']]

df1 = pd.read_csv(teacher)
df1 = df1[['Normal', 'Simple']]

df2 = pd.read_csv(terence)
df2 = df2[['Normal', 'Simple']]

df3 = pd.read_csv(simpitiki)
df3 = df3[['Normal', 'Simple']]

df4 = pd.concat([df, df1, df2, df3])
df4.to_csv(all_together, index=False)