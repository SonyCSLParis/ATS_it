from transformers import pipeline
import csv
from settings import *
import os
import csv

model_checkpoint = "Helsinki-NLP/opus-mt-en-it"
translator = pipeline("translation", model=model_checkpoint)

with open(WIKIPEDIA_DIR + '/normal.training.txt', 'r') as in_compl:
    with open(WIKIPEDIA_DIR + '/simple.training.txt', 'r') as in_semp:

        with open(ENRICHED_DATASET + '/dataset_deepl_PRO_2.csv', 'r') as transl:
            reader = csv.reader(transl)
            head = next(reader)

            f = open('comparison.csv', 'w')

            writer = csv.writer(f)
            writer.writerow(['Normal_ENG', 'Normal_m', 'Normal_Deep','Simple_ENG', 'Simple_m', 'Simple_Deep'])
            lines_compl = in_compl.readlines()
            lines_sempl = in_semp.readlines()

            for i in range(len(lines_compl)):

                out1 = translator(lines_compl[i])
                out11 = out1[0]['translation_text']
                out2 = translator(lines_sempl[i])
                out22 = out2[0]['translation_text']


                line = next(reader)
                compl = line[0]
                sempl = line[1]

                writer.writerow([lines_compl[i], out11, compl, lines_sempl[i], out22, sempl])





