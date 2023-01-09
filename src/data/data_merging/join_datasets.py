from src.settings import *
import pandas as pd

#this is the pacssit dataset without double sentences and correct with respect to the numbers, everything in lower case
pacssit = '/Users/francesca/Desktop/Github/Final_final/output/csv_files/paccssit/paccss_only_1.csv'

#teacher, terence and simpitiki in lower case
teacher = INCOMPLETE_DATASET_DIR + '/teacher_1.csv'
terence = INCOMPLETE_DATASET_DIR + '/terence_1.csv'
simpitiki = INCOMPLETE_DATASET_DIR + '/simpitiki_1.csv'

#path of the joined dataset
all_together =  CSV_FILES_PATH + '/finalized_dataset/finalized_df.csv'


#another phase of joining. We merge our final corpus of 32000 instances (obtained by the join of paccsit, terence, teacher and simpitiki)
#with two dataset translated by the Turk corpus (training and development) and the dataset obtained by the paraller wikipedia corpus version1

finalized_corpus = '/Users/francesca/Desktop/Github/Final_final/output/csv_files/finalized_dataset/finalized_df.csv'
training = '/Users/francesca/Desktop/dataset_utilizzati/training_italiano_pulito.csv'
development = '/Users/francesca/Desktop/dataset_utilizzati/development_italiano_pulito.csv'
wikipedia = '/Users/francesca/Desktop/dataset_utilizzati/dataset_tradotto_pulito.csv'
joined_data = '/Users/francesca/Desktop/Github/Final_final/output/csv_files/augmented/augmented_dataset.csv'


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