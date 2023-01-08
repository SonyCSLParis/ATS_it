from transformers import pipeline
import csv
from nltk.translate.bleu_score import sentence_bleu
import deepl
import evaluate

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-it")
data_path = '/Users/francesca/Desktop/documenti_potenziali/data_1/development.csv'
output_path = '/Users/francesca/Desktop/documenti_potenziali/data_1/development_italiano.csv'

def parsing_turk_corpus(data_input, data_output):

    with open(data_input, 'r') as infile:

        input = csv.reader(infile)
        next(input)

        with open(data_output, 'w') as outfile:
            lista_complesse = []
            lista_semplici = []

            writer = csv.writer(outfile)
            writer.writerow(['Normal', 'Simple'])
            for ele in input:
                if ele[0] not in lista_complesse:

                    complessa = ele[0]
                    semplice = ele[1]
                    lista_complesse.append(complessa)
                    lista_semplici.append(semplice)


                    compl_piped = pipe(complessa)
                    comp_translated = compl_piped[0]['translation_text']
                    meteor = evaluate.load('meteor')
                    results = meteor.compute(predictions=[complessa], references=[comp_translated])
                    print(results['meteor'])
                    if results['meteor'] < 0.1:
                        print(complessa)
                        print(comp_translated)
                    score_c = sentence_bleu([complessa.split()], comp_translated.split())


                    sempl_piped = pipe(semplice)
                    sem_translated = sempl_piped[0]['translation_text']
                    score_s = sentence_bleu([semplice.split()], sem_translated.split())

                    writer.writerow([comp_translated, sem_translated])
            

parsing_turk_corpus(data_path, output_path)


def parsing_wiki(data_input_com, data_input_sem, data_output):
    list_complex_it = []
    list_simple_it = []

    with open(data_input_com, 'r') as in_compl:

        with open(data_input_sem, 'r') as in_semp:

            lines_compl = in_compl.readlines()
            lines_sempl = in_semp.readlines()

            f = open(data_output, 'w')
            # create the csv writer
            writer = csv.writer(f)

            for i in range(len(lines_compl)):
                pip1 = pipe(lines_compl[i])
                pip11 = pip1[0]['translation_text']
                pip2 = pipe(lines_sempl[i])
                pip22 = pip2[0]['translation_text']

                writer.writerow([pip11, pip22])


            f.close()



    with open(data_output, 'w') as outfile:

        writer = csv.writer(outfile)
        writer.writerow(['Normal', 'Simple'])

        for i in range(len(list_complex_it)):
            writer.writerow([list_complex_it[i], list_simple_it[i]])




'''parsing_wiki('/Users/francesca/Desktop/documenti_potenziali/data.v1.split/normal.training.txt',
             '/Users/francesca/Desktop/documenti_potenziali/data.v1.split/simple.training.txt',
             '/Users/francesca/Desktop/documenti_potenziali/data.v1.split/dataset_tradotto.csv')


'''






