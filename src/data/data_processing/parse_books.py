import re
from settings import *
import csv

f = open("/Users/francesca/Desktop/l'ora di punta.txt", 'r')

with open(CSV_FILES_PATH + '/quinta.csv', 'w') as output:
    writer = csv.writer(output)
    writer.writerow(['Normal', 'Simple'])

    for riga in f.readlines():
        res = re.sub(r'(^[^\w]+)|([^\w]+$)', '', riga)
        writer.writerow([res, ' '])



