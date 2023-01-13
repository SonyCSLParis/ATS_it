from settings import *
import pandas as pd

#this is the pacssit dataset without double sentences and correct with respect to the numbers, everything in lower case
pacssit = INCOMPLETE_PROCESSED + '/paccsit_processato.csv'

#teacher, terence and simpitiki in lower case
teacher = INCOMPLETE_PROCESSED + '/teacher_processato.csv'
terence = INCOMPLETE_PROCESSED + '/terence_processato.csv'
simpitiki = INCOMPLETE_PROCESSED + '/simpitiki_processato.csv'

#path of the joined dataset
all_together =  CSV_FILES_PATH + '/finalized_dataset/finalized_df.csv'

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


#another phase of joining. We merge our final corpus of 32000 instances (obtained by the join of paccsit, terence, teacher and simpitiki)
#with two dataset translated by the Turk corpus (training and development) and the dataset obtained by the paraller wikipedia corpus version1

finalized_corpus = CSV_FILES_PATH + '/finalized_dataset/finalized_df.csv'
training = TURKUS_TRANSLATED + '/training_it_processato.csv'
development = TURKUS_TRANSLATED + '/development_it_processato.csv'
wikipedia = WIKIPEDIA_TRANSLATED + '/dataset_tradotto_deepl_processato.csv'
joined_data = CSV_FILES_PATH + '/augmented/augmented_dataset.csv'


#open the dataset and select the columns of interest, eventually join everything
df = pd.read_csv(finalized_corpus)
df = df[['Normal', 'Simple']]

df1 = pd.read_csv(training)
df1 = df1[['Normal', 'Simple']]

df2 = pd.read_csv(development)
df2 = df2[['Normal', 'Simple']]

df3 = pd.read_csv(wikipedia)
df3 = df3[['Normal', 'Simple']]

df4 = pd.concat([df, df1, df2, df3])
df4.to_csv(joined_data, index=False)