from feature_extraction import *
from text import *
from settings import *
import csv

from_path = CSV_FILES_PATH + '/augmented/augmented_dataset.csv'
to_path = CSV_FILES_PATH + '/adaptive/control_token_data.csv'


def generate_feature(path1, path2):

    with open(path1, 'r') as in_file:

        reader = csv.reader(in_file)
        next(reader)

        with open(path2, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(['Normal', 'Simple'])
            for row in reader:
                complex = row[0]
                simple = row[1]

                W = get_word_length_ratio(complex, simple)
                C = get_char_length_ratio(complex, simple)
                LD = round(get_levenshtein_similarity(complex, simple),2)
                WR = round(safe_division(get_lexical_complexity_score(simple),get_lexical_complexity_score(complex)), 2)
                DTD = get_dependency_tree_depth_ratio(complex, simple)

                list_tokens = [W,C,LD,WR,DTD]
                text = ''
                for ele in list_tokens:
                    text += str(ele) + ' '

                addition = 'semplifica: ' + text + complex
                print(addition)
                writer.writerow([addition, simple])

    return

