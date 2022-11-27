import spacy
import csv
import pandas as pd
data_path = '/Users/francesca/Desktop/Github/Final/output/ultimated.csv'
data_output = '/Users/francesca/Desktop/Github/Final/output/processed_ultimated.csv'

nlp = spacy.load('it_core_news_sm')
tokenizer = nlp.tokenizer
list_complex = []
list_simple = []
list_of_cos = []

average_sim = 0.782

#open the complete dataset as a csv
with open(data_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    #read each row
    for row in csv_reader:

        #read each sentence of the raw
        for i in range(len(row)):

            #if we are dealing with the complex sentences
            if i == 0:

                #create the doc
                doc1 = nlp(row[i])

                #filter out stop_words, punctuation and non-latin characters
                frase = [token for token in doc1 if token.is_punct == False and token.is_ascii == True and token.is_stop == False]

                #Rebuild a list of strings
                frase_1 = [str(token) for token in frase]

                #if the string is longer than 0, recreate the string
                if len(frase_1) > 0:
                    final_1 = ' '.join(frase_1)

                # otherwise assign None to it
                else:
                    final_1 = None


            #if we are dealing with the simple sentences
            if i == 1:

                #create the doc
                doc2 = nlp(row[i])

                # filter out stop_words, punctuation and non-latin characters
                frase2 = [token for token in doc2 if token.is_punct == False and token.is_ascii == True and token.is_stop == False]

                # rebuild a list of strings
                frase_2 = [str(token) for token in frase2]

                # if the string is longer than 0, recreate the string
                if len(frase_2) > 0:
                    final_2 = ' '.join(frase_2)

                # otherwise assign None to it
                else:
                    final_2 = None

        #if both the sentences are different from None, I calculate their cosine similarity
        if final_1 != None and final_2 != None:
            d1 = nlp(final_1)
            d2 = nlp(final_2)
            similarity = d1.similarity(d2)
            list_of_cos.append(d1.similarity(d2))

        #I set the value to 0.7 because the average cosine similarity is 0.78
        if similarity > 0.7:

            #this time I filter out only for puntuation and non-latin characters
            to_charge = [token for token in doc1 if token.is_punct == False and token.is_ascii == True]

            # I rebuild the list of strings
            to_charge1 = [str(token) for token in to_charge]

            #if the list of string is longer than 2
            if len(to_charge1) > 2:

                # I recreate the string
                f = ' '.join(to_charge1)

                # append it to the list of complex sentences
                list_complex.append(f)

                #do the same thing for the simplified sentence
                to_charge_sim = [token for token in doc2 if token.is_punct == False and token.is_ascii == True]
                to_charge1_sim = [str(token) for token in to_charge_sim]

                f_sim = ' '.join(to_charge1_sim)
                list_simple.append(f_sim)





#I create the final dataset and I save ita s a csv
d = {'Normal':list_complex,'Simple':list_simple}
df = pd.DataFrame(d)
df.to_csv(data_output,index=False)



















