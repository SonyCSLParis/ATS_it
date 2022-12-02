import spacy
import csv
import pandas as pd
data_path = '/Users/francesca/Desktop/Github/Final/output/ultimated.csv'
data_output = '/Users/francesca/Desktop/Github/Final/output/lemmatized_ultimated.csv'

nlp = spacy.load('it_core_news_sm')


docum = nlp('Il galleggiante rimaneva a galla e non più permetteva di andare a mangià la pizza, questo eè un peccato')

for tok in docum:
    print(tok.text, tok.is_ascii)