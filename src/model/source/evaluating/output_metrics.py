import csv
from settings import *

def average_evaluation(input_csv):
    metrics = ['sari', 'meteor', 'rouge', 'bleu']
    sari = 0
    meteor = 0
    rouge = 0
    bleu = 0

    with open(input_csv, 'r') as in_file:

        reader = csv.reader(in_file)
        next(reader)

        i = 0
        for row in reader:
            i +=1

            sari += float(row[2])
            meteor += float(row[3])
            rouge += float(row[4])
            bleu += float(row[5])

    to_report = [(metrics[0],sari//i), (metrics[1],round(meteor/i, 3)), (metrics[2],round(rouge/i,3)),(metrics[3] ,bleu//i) ]
    return to_report

print(average_evaluation(CSV_EVAL_OUTPUT + '/augmented_20.csv'))
