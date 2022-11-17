import csv
from src.data_merging.settings import *
import re


full_dir = DATA_DIR + 'simpitiki-v1.txt'

with open(full_dir, 'r') as infile:
    original = []
    simple = []

    for line in infile:

        if 'before' in line:
            riga = line

            if '&lt;del&gt;' in riga:
                riga = re.sub('&lt;del&gt;', '', riga)

            if '&lt;/del&gt;' in riga:
                riga = re.sub('&lt;/del&gt;', '', riga)

            if '&#xF2' in riga:
                riga = re.sub('&#xF2', 'ó', riga)

            if '&#xE8;' in riga:
                riga = re.sub('&#xE8;', 'è', riga)

            if '&#xE0' in riga:
                riga = re.sub('&#xE0', 'à', riga)

            if '&#xF4;' in riga:
                riga = re.sub('&#xF4;', 'o', riga)

            if '[...]' in riga:
                riga = riga.replace('[...]', '')



            start = riga.find('e>')
            end = riga.find('</')
            origi = riga[start + 2:end]
            original.append(origi.lower())


        if 'after' in line:

            riga = line

            if '&lt;ins&gt;' in riga:
                riga = re.sub('&lt;ins&gt;', '', riga)

            if '&lt;/ins&gt;' in riga:
                riga = re.sub('&lt;/ins&gt;', '', riga)

            if '&#xF2' in riga:
                riga = re.sub('&#xF2', 'ó', riga)

            if '&#xE8;' in riga:
                riga = re.sub('&#xE8;', 'è', riga)

            if '&#xE0' in riga:
                riga = re.sub('&#xE0', 'à', riga)

            if '&#xF4;' in riga:
                riga = re.sub('&#xF4;', 'o', riga)

            if '[...]' in riga:
                riga = riga.replace('[...]', '')

            start = riga.find('r>')
            end = riga.find('</')
            simp = riga[start + 2:end]
            simple.append(simp.lower())


#eliminate copies of the same phrase
dizio = {}

for i in range(len(original)):
    if original[i] not in dizio:
        dizio[original[i]] = simple[i]


with open(INTERMEDIATE_DIR + 'simpitiki.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    header = ['index','Sentente_1', 'Sentence_2']
    writer.writerow(header)
    lista = list(dizio.items())
    for i in range(len(lista)):
        if len(lista[i][0]) > 15:
            writer.writerow((i, lista[i][0], lista[i][1]))








