from nltk.corpus import wordnet
import spacy
import csv
from datasets import load_dataset
from transformers import pipeline

nlp = spacy.load('it_core_news_sm')

#synonym generation
def lexical_augmentation(input_file, output_file):

    with open(input_file, 'r') as inputfile:
        reader = csv.reader(inputfile)
        header = next(reader)

        with open(output_file, 'w') as outputfile:
            writer = csv.writer(outputfile)
            for riga in reader:
                parallel = []
                for sentence in riga:
                    syn_sentence = ''
                    for word in sentence.split():
                        try:
                            ll = wordnet.synsets(word, lang='ita')
                            lista = [synset.lemmas(lang='ita') for synset in ll]
                            name= [hyper[0].name().split('.')[0] for hyper in lista]
                            syn = name[0]
                            d1 = nlp(word)
                            d2 = nlp(syn)
                            cs = d1.similarity(d2)
                            syn_word = syn if cs >= 0.6 else word
                        except:
                            syn_word = word
                        to_append = syn_word.replace('_', ' ') + ' '
                        syn_sentence += to_append

                    parallel.append(syn_sentence)
                writer.writerow(parallel)
    return


#back-and-forth translation

def back_and_forth(file_to_parse, file_to_generate):
    pipe_back = pipeline("translation", model="Helsinki-NLP/opus-mt-it-de")
    pipe_forth = pipeline("translation", model="Helsinki-NLP/opus-mt-de-it")

    with open(file_to_parse, 'r') as input_file:

        reader = csv.reader(input_file)

        #skip header
        next(reader)
        with open(file_to_generate, 'w') as output_file:

            writer = csv.writer(output_file)
            writer.writerow(['Normal', 'Simple'])

            for row in reader:

                org = row[0]
                spl = row[1]

                org_back = pipe_back(org)
                org_forth = pipe_forth(org_back[0]['translation_text'])

                spl_back = pipe_back(spl)
                spl_forth = pipe_forth(spl_back[0]['translation_text'])

                writer.writerow([org_forth[0]['translation_text'], spl_forth[0]['translation_text']])

    return


file1 = '/Users/francesca/Desktop/Github/Final_final/output/csv_files/ter_tea_sim/tts.csv'
file2 = '/Users/francesca/Desktop/Github/Final_final/output/csv_files/ter_tea_sim/tts_augmented_back_and_forth.csv'
back_and_forth(file_to_parse = file1, file_to_generate = file2)
