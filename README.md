# Automatic Text Simplification for SENDs 
### Working with a low-resource language, such as Italian

The following project aims to use a state-of-the-art model, a transformer to be more precise, within the discipline of Natural Language Processing, in order to fine-tune our task of interest, i.e. to create text simplifications to enable better reading for individuals, both children and adults, who suffer from text comprehension disorder. It also includes the pre-processing of the datasets used and the code used and implemented to parse the original files containing the parallel sentences (Complex and Simple) and merge all available resources. Finally, attempts were made at data augmentation, the code for which is given below. 

## Contributors
Francesca Padovani, trainee at CSL Sony Paris contributed to the creation of this repository, under the supervision of Martina Galletti.

## Structure
The repository contains the **src** code, the structure of which will be explained below, the **requirements.txt** file, which contains all the libraries needed to set up your environment and be able to run the code, and the **settings.py** file in which we placed all local paths and assigned them to global variables, which we then reused in the scripts to access data. All the input datasets, the pre-processed ones and those generated in the Hugging Face format needed to tune the model, were not published, partly because of the large size. The online sources from which I primarily retrieved the three corpus used in this project are as follows: 

- [PACCSS-IT (Parallel Corpus of Complex-Simple Sentences for ITalian)](http://www.italianlp.it/resources/paccss-it-parallel-corpus-of-complex-simple-sentences-for-italian/)

- [TERENCE and TEACHER](http://www.italianlp.it/resources/terence-and-teacher/)

- [SIMPITIKI](https://github.com/dhfbk/simpitiki)


#### Data 
#### Model Folder
#### Dependencies & Python Version 
#### requirements.txt 

## How to run the code and train a model

### License
