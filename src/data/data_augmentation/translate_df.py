import csv
import deepl
from settings import *
import os

def cs_turk_corpus(data_input, data_output):
    '''
    This function is intended to calculate the cosine similarity between the original and simplified sentences in TURKUS CORPUS
    and create a new .csv file in which these values are also inserted in the column.
    :param data_input: directory of the input file that should be processed
    :param data_output: directory of the output file that should be generated
    :return: a new .csv file
    '''

    with open(data_input, 'r') as infile:

        input = csv.reader(infile)
        next(input)

        with open(data_output, 'w') as outfile:

            writer = csv.writer(outfile)
            writer.writerow(['Normal', 'Simple', 'Similarity'])
            for ele in input:

                difficult = ele[0]
                easy = ele[1]

                c_sim = nlp(difficult).similarity(nlp(easy))
                writer.writerow([difficult, easy, c_sim])
    return


def parse_turk_corpus(data_input, data_output):
    '''
    This function allows you to parse the file containing the simplifications and similarities and go to select one simplification, from the 10 proposed ones.
    If the cosine similarity among the 10 sentences does not have a consistent range of difference then take the two sentences that have a lower cosine similarity,
    otherwise prefer the sentence that has the cosine similarity in the middle among all the others.
    After choosing the preferred sentence, then translate it with the DeepL API and create a new .csv file containing the translations.

    :param data_input: directory of the input file that should be processed
    :param data_output: directory of the output file that should be generated
    :return: a new .csv file
    '''
    auth_key = AUTHENTICATION_DEEPL

    with open(data_input, 'r') as infile1:

        input1 = csv.reader(infile1)
        next(input1)

        set_complex = []

        for ele in input1:

            #I save only the unique phrases
            if ele[0] not in set_complex:
                set_complex.append(ele[0])

        #generate a list of lists both for all the simplifications and for the cosine similairties,
        #these inner lists being as many as the amount of unique sentences in the corpus
        temporary = [[] for i in range(len(set_complex) + 1)]
        css = [[] for i in range(len(set_complex) + 1)]

        with open(data_input, 'r') as infile2:

            input2 = csv.reader(infile2)
            next(input2)

            keep_tracks = []

            i = 0

            for ele in input2:
                if ele[0] not in keep_tracks:
                    i +=1

                    temporary[i].append(ele[1])
                    css[i].append(ele[2])
                    keep_tracks.append(ele[0])


                if ele[0] in keep_tracks:
                    temporary[i].append(ele[1])
                    css[i].append(ele[2])

        with open(data_output, 'w') as outfile:

            writer = csv.writer(outfile)
            writer.writerow(['Normal', 'Simple'])

            for i in range(len(temporary)):

                if i == 0:
                    pass

                else:
                    #check the entity of the range
                    range_cs = float(max(css[i])) - float(min(css[i]))
                    tuple = [(css[i][j], temporary[i][j]) for j in range(len(css[i]))]
                    newList = sorted(tuple, key=lambda x: x[0])

                    #if the range is bigger than 0.2
                    if range_cs > 0.2:

                        #choose the simplification which has the average cosine similarity among the others
                        moyen = len(newList) // 2
                        choice_simple = newList[moyen][1]
                        choice_complex = keep_tracks[i -1]

                    else:
                        #choose the simplification which has the smaller cosine similarity
                        choice_simple = newList[0][1]
                        choice_complex = keep_tracks[i -1]

                    #instantiate the translator and proceed with the actual translations
                    translator = deepl.Translator(auth_key)
                    trans_1 = nlp(str(translator.translate_text(choice_complex, target_lang="IT")))
                    trans_2 = nlp(str(translator.translate_text(choice_simple, target_lang="IT")))
                    writer.writerow([trans_1, trans_2])

    return


#procedure for the TURKUS corpus
base_dir_sim = TURKUS_COS_SIM
base_dir = TURKUS_DIR
if not os.path.exists(base_dir_sim):
    os.mkdir(base_dir_sim)
    for file in os.listdir(base_dir):
        if file != 'README':
            input_path = base_dir + '/' + file
            output_path = path + '/' + file[:-4] + '_similarity.csv'
            cs_turk_corpus(data_input=input_path, data_output=output_path)

for file in os.listdir(base_dir_sim):
    i = file.find('_')
    name = file[:i+1] + 'it.csv'
    input_dir = str(path) + '/' + file
    outpur_dir = TURKUS_TRANSLATED + '/' + name

    parse_turk_corpus(data_input= input_dir, data_output=outpur_dir)



def parsing_wiki(data_input_com, data_input_sem, data_output):
    '''
    This function allows the translation of sentences in the Wikipedia corpus for the English language.
    We always use the API of DeepL and save the translations to a new .csv file.

    :param data_input_com: directory of the input file that should be processed; it contains complex sentences
    :param data_input_sem: directory of the input file that should be processed; it contains simple sentences
    :param data_output: directory of the output file that should be generated; containing the translations
    :return: a new .csv file containing teh translations
    '''
    auth_key = AUTHENTICATION_DEEPL

    with open(data_input_com, 'r') as in_compl:

        with open(data_input_sem, 'r') as in_semp:

            lines_compl = in_compl.readlines()
            lines_sempl = in_semp.readlines()

            f = open(data_output, 'w')

            writer = csv.writer(f)
            writer.writerow(['Normal', 'Simple'])

            for i in range(len(lines_compl)):
                if i < 1000:
                    translator = deepl.Translator(auth_key)
                    c = translator.translate_text(lines_compl[i], target_lang= 'IT')
                    s = translator.translate_text(lines_sempl[i], target_lang= 'IT')
                    writer.writerow([c, s])



            f.close()

    return



















