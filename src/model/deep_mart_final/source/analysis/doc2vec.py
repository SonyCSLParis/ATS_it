import spacy
from gensim.models.doc2vec import Doc2Vec
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile


#here a class ToVec is created and implemented
class ToVec:
    def __init__(self, model_path: str):

        #I give the path of the already trained model_deep to be opened
        self.model = Doc2Vec.load(model_path)

        #I load the italian model_deep for spacy library
        self.nlp = spacy.load("it_core_news_sm")

    def cosine_distance(self, normal_str: str, simple_str: str) -> float:
        normal_vector = self.model.infer_vector(normal_str.split())
        simple_vector = self.model.infer_vector(simple_str.split())
        return spatial.distance.cosine(normal_vector, simple_vector)

    def cosine_similarity(self, normal_string: str, simple_string: str) -> float:
        normal_vector = self.model.infer_vector(normal_string.split())
        simple_vector = self.model.infer_vector(simple_string.split())
        similarity = cosine_similarity([normal_vector], [simple_vector])
        return similarity[0][0]

    def spacy_cosine_similarity(self, normal_string: str, simple_string: str) -> float:
        normal_vector = self.nlp(normal_string)
        simple_vector = self.nlp(simple_string)
        similarity = cosine_similarity([normal_vector], [simple_vector])
        return similarity[0][0]

    def export_w2vec_data(self, save_file_path: str):
        self.model.wv.save_word2vec_format(save_file_path)


#I took this function from the documentation of Gensim, it is an helper function which allows to load the documents in the two cases when
#we read the train and test data

def read_corpus(fname, tokens_only=False):
    with open(fname, 'r') as f:
        #skip header
        next(f)
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

#indeed here we load the train and test corpus in two different ways, activating the flag, tokens_only
train_corpus = list(read_corpus('/Users/francesca/Desktop/deep_mart_final/data/df_train_ultimated.csv'))
test_corpus = list(read_corpus('/Users/francesca/Desktop/deep_mart_final/data/df_test_ultimated.csv', tokens_only=True))

#here I instantiate my model_deep (randomly and we should check and investigate how we can do it in a suitable way)
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

#I build the vocabulary, according to our model_deep
model.build_vocab(train_corpus)

#and at the end I train my model_deep on the training corpus
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

#I open a temporary file in order to save the trained Gensim Doc2Vec model_deep
fname = get_tmpfile("/Users/francesca/Desktop/deep_mart_final/model_deep/doc_2_vec_model/modello2")

model.save(fname)