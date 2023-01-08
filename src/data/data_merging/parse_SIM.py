import csv
from settings import *
import re


#directory of the original .txt file, downloaded from the official Github of the project https://github.com/dhfbk/simpitiki
full_dir = DATA_DIR + '/simpitiki-v2.txt'

#function which allows to write to file the collected parallel sentences
def write_on_file(output_data, dizionario):
    with open(INCOMPLETE_DATASET_DIR + output_data, 'w') as outfile:
        writer = csv.writer(outfile)
        header = ['index','Sentence_1', 'Sentence_2']
        writer.writerow(header)
        lista = list(dizionario.items())
        for i in range(len(lista)):
            if not len(lista[i][0]) < 20 and not len(lista[i][1]) < 20:
                writer.writerow((i, lista[i][0], lista[i][1]))

#we parse and clean the text
with open(full_dir, 'r') as infile:
    original = []
    simple = []
    dizionario = []

    for line in infile:

        if 'before' in line:
            aggiungi = []
            riga = line

            if '&lt;del&gt;' in riga:
                riga = re.sub('&lt;del&gt;', '', riga)


            if '&lt;/del&gt;' in riga:
                riga = re.sub('&lt;/del&gt;', '', riga)

            if '&#xF2' in riga:
                riga = re.sub('&#xF2;', 'ó', riga)

            if '&#xE8;' in riga:
                riga = re.sub('&#xE8;', 'è', riga)

            if '&#xE0' in riga:
                riga = re.sub('&#xE0;', 'à', riga)

            if '&#x94' in riga:
                riga = re.sub('&#x94;', '', riga)

            if '&#x93' in riga:
                riga = re.sub('&#x93;', '', riga)

            if '&#xF4;' in riga:
                riga = re.sub('&#xF4;', 'o', riga)

            if '&#xec;' in riga:
                riga = re.sub('&#xec;', 'i', riga)

            if '[...]' in riga:
                riga = riga.replace('[...]', '')



            start = riga.find('re>')
            end = riga.find('</b')
            origi = riga[start + 3:end]
            aggiungi.append(origi)




        if 'after' in line:
            riga = line

            if '&lt;ins&gt;' in riga:
                riga = re.sub('&lt;ins&gt;', '', riga)

            if '&lt;/ins&gt;' in riga:
                riga = re.sub('&lt;/ins&gt;', '', riga)

            if '&#xF2' in riga:
                riga = re.sub('&#xF2;', 'ó', riga)

            if '&#xE8;' in riga:
                riga = re.sub('&#xE8;', 'è', riga)

            if '&#xE0' in riga:
                riga = re.sub('&#xE0;', 'à', riga)

            if '&#x94' in riga:
                riga = re.sub('&#x94;', '', riga)

            if '&#x93' in riga:
                riga = re.sub('&#x93;', '', riga)

            if '&#xF4;' in riga:
                riga = re.sub('&#xF4;', 'o', riga)

            if '[...]' in riga:
                riga = riga.replace('[...]', '')

            start = riga.find('er>')
            end = riga.find('</a')
            simp = riga[start + 3:end]
            aggiungi.append(simp)

            if len(aggiungi) == 2:
                original.append(aggiungi[0])
                simple.append(aggiungi[1])

dizio = {}
for i in range(len(original)):
    if original[i] != '' and original[i] not in dizio:
        d1 = nlp(original[i])
        d2 = nlp(simple[i])
        cs = d1.similarity(d2)
        if cs > 0.7:
            dizio[original[i]] = simple[i]


write_on_file('/simpitiki_1.csv', dizionario=dizio)







