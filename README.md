# Automatic Text Simplification for SENDs 
### Working with a low-resource language, such as Italian

The following project aims to use a state-of-the-art model, a transformer to be more precise, within the discipline of Natural Language Processing, in order to fine-tune our task of interest, i.e. to create text simplifications to enable better reading for individuals, both children and adults, who suffer from text comprehension disorder. It also includes the pre-processing of the datasets used and the code used and implemented to parse the original files containing the parallel sentences (Complex and Simple) and merge all available resources. Finally, attempts were made at data augmentation, the code for which is given below. 

## Contributors
Francesca Padovani, trainee at CSL Sony Paris contributed to the creation of this repository, under the supervision of Martina Galletti.

## Structure
The repository contains the **src** code, the structure of which will be explained below, the **requirements.txt** file, which contains all the libraries needed to set up your environment and be able to run the code, and the **settings.py** file in which we placed all local paths and assigned them to global variables, which we then reused in the scripts to access data. All the input datasets, the pre-processed ones and those generated in the Hugging Face format needed to tune the model, were not published, partly because of the large size. The online sources from which I primarily retrieved the three corpus used in this project are as follows: 

- [PACCSS-IT (Parallel Corpus of Complex-Simple Sentences for ITalian)](http://www.italianlp.it/resources/paccss-it-parallel-corpus-of-complex-simple-sentences-for-italian/): PaCCSS-IT is a corpus of Complex-Simple Aligned Sentences for ITalian of about 63,000 pairs of sentences extracted from the ItWaC corpus, the largest copy-right free corpus of contemporary Italian web texts. 

- [TERENCE and TEACHER](http://www.italianlp.it/resources/terence-and-teacher/): Terence and Teacher are two corpora of original and manually simplified texts aligned at sentence level. Terence contains 32 short Italian novels for children and their manually simplified version carried out by experts (linguists and psycholinguists) targeting children with problems in text comprehension. The corpus comprises 1036 original and 1060 simplified sentences. Teacher is a corpus of 18 pairs of Italian documents belonging to different genres (e.g. literature, handbooks) and used in educational settings. They contain both the original text and their manually simplified version mainly carried out by teachers, for a total of 266 original and 255 simplified sentences.

- [SIMPITIKI](https://github.com/dhfbk/simpitiki): SIMPITIKI is a Simplification corpus for Italian and it consists of two sets of simplified pairs: the first one is harvested from the Italian Wikipedia in a semi-automatic way; the second one is manually annotated sentence-by-sentence from documents in the administrative domain.


### src 
Inside the src folder you can find three sub-categories named: **analysis**, **data_merging**, **model** that relate to three different stages of the project.

#### data merging
This folder is quite composite:

- the two scripts `parse_TT.py` and `parse_SIM.py` used to parse both the folders in which the dataset sentences (Terence and Teacher) were stored and the .txt file in which the original and simplified Simpitiki sentences were contained;

- the script `open_paccsit.py` where PACCSS-IT is parsed and the parallel sentences are collected;

- the script `join_datasets.py` allows you to merge the three individual datasets into a final one;

- the two scripts `clean_text.py` and `preprocess_paccsit.py` are used for pre-processing analyses to clean up datasets, either lemmatising, removing stopwords, or filtering for cosine similarity between complex and simple sentences. Various tests and trials were conducted during the project.

#### analysis 
This folder contains scripts for doing quantitative analyses and making qualitative judgements (as far as possible) on the datasets used. 
These are simple *NLP* analyses and the main library used is *spacy*. By running `main.py`, one can obtain graphs and metrics, including the distribution of Part Of Speech, see how many simple structures (SVO - Subject Verb Object -) are present in both simple and complex sentences, and understand the Mood and Tense of verbs. 


#### Model Folder
#### Dependencies & Python Version 
#### requirements.txt 

## How to run the code and train a model

### License
