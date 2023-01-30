import csv
from settings import *

#let the matrics be saved on a json file or a csv file
def average_evaluation(input_csv):
    metrics = ['sari', 'meteor', 'rouge', 'bleu']
    sari = 0
    meteor = 0
    rouge = 0
    bleu = 0

    sari1 = 0
    meteor1 = 0
    rouge1 = 0
    bleu1 = 0

    with open(input_csv, 'r') as in_file:

        reader = csv.reader(in_file)
        next(reader)

        i = 0
        j = 0
        for row in reader:
            i +=1
            j +=1

            sari += float(row[2])
            meteor += float(row[3])
            rouge += float(row[4])
            bleu += float(row[5])

            sari1 += float(row[7])
            meteor1 += float(row[8])
            rouge1 += float(row[9])
            bleu1 += float(row[10])



    to_report_first = [(metrics[0],sari//i), (metrics[1],round(meteor/i, 3)), (metrics[2],round(rouge/i,3)),(metrics[3] ,bleu//i)]
    to_report_second = [(metrics[0],sari1//i), (metrics[1],round(meteor1/i, 3)), (metrics[2],round(rouge1/i,3)),(metrics[3],bleu1//i)]
    return to_report_first, to_report_second

print(average_evaluation('/Users/francesca/Desktop/Github/PROJECT_SONY/src/model/model_deep/csv_output/adaptive_8_params.csv'))
