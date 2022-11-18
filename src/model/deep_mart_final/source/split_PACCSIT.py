import pandas as pd
import csv
from settings import *

# with the following code I splitted the original csv into two sets (training set containing 45000
# aligned sentences and the test set containing 16785 aligned sentences)
df = pd.read_csv(OUTPUT_DIR + 'ultimated.csv')
dataset = OUTPUT_DIR + 'ultimated.csv'
filename1 = OUTPUT_DIR + 'df_train_ultimated.csv'
filename2 = OUTPUT_DIR + 'df_test_ultimated.csv'

with open(dataset, 'r') as infile:
    csvreader = csv.reader(infile)
    header = next(csvreader)
    #header = header[1:]
    data1 = []
    data2 = []
    i = 0

    for row in csvreader:
        #row1 = row[1:]
        row1 = row[:]
        if i < 55000:
            data1.append(row1)
            i += 1

        else:
            data2.append(row1)
            i += 1

    with open(filename1, 'w', newline="") as outfile1:
        csvwriter = csv.writer(outfile1)
        csvwriter.writerow(header)
        csvwriter.writerows(data1)

    with open(filename2, 'w', newline="") as outfile2:
        csvwriter = csv.writer(outfile2)
        csvwriter.writerow(header)
        csvwriter.writerows(data2)
