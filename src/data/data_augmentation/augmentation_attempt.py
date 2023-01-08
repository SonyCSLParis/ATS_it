from nltk.corpus import wordnet
import spacy
import csv
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


#back-and-forth translation
