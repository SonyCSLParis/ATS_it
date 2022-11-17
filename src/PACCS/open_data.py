import csv

import pandas as pd

from settings import *


def open_txt():
    # Open the corpus and start parsing it
    with open(DATA_DIR + '/PACCSS-IT.txt', "r") as infile:

        strip_ = (line.strip() for line in infile)
        lines_splitted = (line.split("? ") if '?' in line else line.split('. ') for line in strip_)

        with open(OUTPUT_DIR + '/final.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            corpus = []
            err_count = 0
            for line in lines_splitted:
                lung = len(line)
                # if the length of the list is already 7, I immediately write the line in the output file
                if lung == 7:
                    writer.writerow(line)
                    corpus.append(line)
                # if the length is different than 7
                elif lung != 7:
                    # print(lung)
                    # print()
                    # print(line)

                    new_line = []
                    # if in that string is not present a '\t' add it immediately on the new_line (new list we are creating)
                    for instance in line:
                        if not '\t' in instance:
                            new_line.append(instance)

                        else:
                            # otherwise split the string where you find a tag, iterate through it and append the single string into the new_line
                            building = instance.split('\t')
                            for ins in building:
                                new_line.append(ins)

                    new1 = [ele.strip() for ele in new_line if not ele == '']

                    if len(new1) == 7:
                        writer.writerow(new1)
                        corpus.append(new1)

                    else:
                        # after manually inspecting the lists I saw that there were very short and small strings such as ('..', 'L', 'Pag', 'Art', 'poi ...', '...'
                        # therefore I filtered for the length of the string in order to keep only longer strings
                        new2 = []
                        for ele in new1:
                            if len(ele) > 12:
                                new2.append(ele)
                        if len(new2) == 7:
                            writer.writerow(new2)
                            corpus.append(new2)
                        else:
                            err_count += 1

        # Some lines (rows) where not properly designed and build, therefore we discarded them and print how many of them were not considered
        print(f'The wrong lines are: {err_count}.')

        # Reassign the column names (header)
        final_corpus = pd.DataFrame(corpus)
        final_corpus = final_corpus.rename(
            {0: 'Sentence_1', 1: 'Sentence_2', 2: 'Cosine_Similarity', 3: 'Confidence', 4: 'Readability_1',
             5: 'Readability_2',
             6: '(Readability_1-Readability_2)'}, axis=1)

        return final_corpus
