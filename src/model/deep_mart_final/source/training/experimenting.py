import os
import openai
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from transformers import EncoderDecoderModel, AutoTokenizer
'''OPENAI_API_KEY = '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'
# Load your API key from an environment variable or secret management service
openai.api_key = OPENAI_API_KEY

response = openai.Completion.create(model="text-davinci-003", prompt="Simplify the following code: I have been to the supermarket and I bought more than needed. I will probably contribute to food waste in the next weeks. ", temperature=0.8, max_tokens=30)


text = 'Questa sera pulisco bene la casa e poi metto delle trappole per topo'
aug = naw.SynonymAug(aug_src='wordnet', lang= 'ita')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")

print(augmented_text)



aug = naw.RandomWordAug(action="swap")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)



aug = naw.RandomWordAug()
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)'''

from transformers import pipeline
import csv
import deepl


pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-it")

data_path = '/Users/francesca/Desktop/documenti_potenziali/data_1/development.csv'
output_path = '/Users/francesca/Desktop/documenti_potenziali/data_1/development_italiano.csv'

def parsing_turk_corpus(data_input, data_output):

    with open(data_input, 'r') as infile:

        input = csv.reader(infile)
        header = next(input)

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
                    sempl_piped = pipe(semplice)
                    sem_translated = sempl_piped[0]['translation_text']

                    writer.writerow([comp_translated, sem_translated])
            

#parsing_turk_corpus(data_path, output_path)


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




parsing_wiki('/Users/francesca/Desktop/documenti_potenziali/data.v1.split/normal.training.txt',
             '/Users/francesca/Desktop/documenti_potenziali/data.v1.split/simple.training.txt',
             '/Users/francesca/Desktop/documenti_potenziali/data.v1.split/dataset_tradotto.csv')









