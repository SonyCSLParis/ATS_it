import csv
import os
import re
from src.data_merging.settings import *


full_dir = DATA_DIR
original_teach = []
semplici_teach = []
original_ter = []
semplici_ter = []
for filename in os.listdir(full_dir):
    if filename == 'Teacher':
        part_1 = '/Teacher'
        for filen in os.listdir(full_dir + part_1):
            if filen.endswith('.txt'):
                entire_dir = full_dir + part_1 + '/' + filen
                with open(entire_dir, 'r') as infile:
                    original_p = []
                    semplici_p = []
                    for line in infile:

                        if not 'doc id' in line and not '/doc' in line and line != '':

                            if 'frase_all' in line:

                                start_id = line.find('id=')
                                end_id = line.find('" ')
                                start_all = line.find('l=')
                                end_all = line.find('>')
                                end_sent = line.find('</')
                                id, frase_all, frase = line[start_id + 4: end_id], line[
                                                                                   start_all + 3: end_all - 1], line[
                                                                                                                end_all + 1: end_sent]
                                original_p.append((id, frase_all, frase))


                            else:
                                start_id = line.find('id=')
                                end_id = line.find('>')
                                # start_sent = line.find('">')
                                end_sent = line.find('</')
                                id, frase = line[start_id + 4: end_id - 1], line[end_id + 1: end_sent]
                                if id == '':
                                    id = line[start_id + 3: end_id - 1]
                                semplici_p.append((id, frase))

                    for i in range(len(original_p)):
                        if original_p[i][0] == original_p[i][1] and original_p[i][1] != '':
                            for ele in semplici_p:
                                if ele[0] == original_p[i][0]:
                                    orig_p1 = re.sub('"', '', original_p[i][2])
                                    seml_p1 = re.sub('"', '', ele[1])
                                    original_teach.append(orig_p1)
                                    semplici_teach.append(seml_p1)


                        else:
                            for ele in semplici_p:
                                if ele[0] == original_p[i][1]:
                                    orig_p1 = re.sub('"', '', original_p[i][2])
                                    seml_p1 = re.sub('"', '', ele[1])
                                    original_teach.append(orig_p1)
                                    semplici_teach.append(seml_p1)

    with open(INTERMEDIATE_DIR + 'teacher.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        header = ['index','Sentence_1', 'Sentence_2']
        writer.writerow(header)
        for i in range(len(original_teach)):
            if semplici_teach[i] != '':
                writer.writerow([str(i), original_teach[i], semplici_teach[i]])



    if filename == 'Terence':
        part_2 = '/Terence'
        for filen in os.listdir(full_dir + part_2):
            if filen != '.DS_Store':
                for fn in os.listdir(full_dir + part_2 + '/' + filen):
                    if fn.endswith('.txt'):
                        entire_dir = full_dir + part_2 + '/' + filen + '/' + fn
                        with open(entire_dir, 'r') as infile:
                            original_p = []
                            semplici_p = []
                            for line in infile:
                                if not 'semplificato' in line and not 'originale' in line and not 'doc id' in line and not '</doc>' in line:

                                    if 'frase_al' in line:

                                        start_id = line.find('id=')
                                        end_id = line.find('" ')
                                        start_all = line.find('l=')
                                        end_all = line.find('>')
                                        end_sent = line.find('</')
                                        id, frase_all, frase = line[start_id + 4: end_id], line[
                                                                                           start_all + 3: end_all - 1], line[end_all + 1: end_sent]

                                        original_p.append((id, frase_all, frase))


                                    else:
                                        start_id = line.find('id=')
                                        end_id = line.find('>')
                                        # start_sent = line.find('">')
                                        end_sent = line.find('</')
                                        id, frase = line[start_id + 4: end_id - 1], line[end_id + 1: end_sent]
                                        if id == '':
                                            id = line[start_id + 3: end_id - 1]

                                        semplici_p.append((id, frase))

                            for i in range(len(original_p)):
                                if original_p[i][0] == original_p[i][1] and original_p[i][1] != '':
                                    for ele in semplici_p:
                                        if ele[0] == original_p[i][0]:
                                            original_ter.append(original_p[i][2])
                                            semplici_ter.append(ele[1])


                                else:
                                    for ele in semplici_p:
                                        if ele[0] == original_p[i][1]:
                                            original_ter.append(original_p[i][2])
                                            semplici_ter.append(ele[1])

    with open( INTERMEDIATE_DIR + 'terence.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        header = ['index','Sentence_1', 'Sentence_2']
        writer.writerow(header)
        for i in range(len(original_ter)):
            if semplici_ter[i] != '':
                writer.writerow([str(i), original_ter[i], semplici_ter[i]])




