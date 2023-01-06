# Automatic Text Simplification for SENDs 
### Working with a low-resource language, such as Italian

The following project aims to use a state-of-the-art model, a transformer to be more precise, within the discipline of Natural Language Processing, in order to fine-tune our task of interest, i.e. to create text simplifications to enable better reading for individuals, both children and adults, who suffer from text comprehension disorder. It also includes the pre-processing of the datasets used and the code used and implemented to parse the original files containing the parallel sentences (Complex and Simple) and merge all available resources. Finally, attempts were made at data augmentation, the code for which is given below. [Link to the web page of the Project](https://csl.sony.fr/project/ai-for-send-students/)

## Contributors

- Francesca Padovani
- Martina Galletti

## Dependencies & Python Version 
In the repository you can find the `requirements.txt` file, which contains all the libraries needed to set up your environment and be able to run the code.  This project requires `Python 3.10.6` version to be run.

## How to run the code and train a model
In order to use the Hugging Face pipeline to fine-tune your model you need to provide a dataset which should be an instance of the `HuggingFaceDataset` class. It needs to have one column with the normal version (`Normal`) and one for the simplified version (`Simple`). 

To train a model you then simply run something like:

`python /your/path/to/deep-martin/src/hf_transformer_trainer.py \
--num_train_epochs 20  # number of the epochs  
--ds_path /path/ # path to you dataset.
--save_model_path /path/  # path to where the trained model should be stored.
--pretrained_model_path /path/ # path where your pre-trained checkpoints are saved
--training_output_path /path/  # path to where the checkpoints and the training data should be stored.
--tokenizer_id /tokenizer/ # Path or identifier to Hugging Face tokenizer.`

To evaluate your model you will find instruction in the function within the `evaluator.py` script. You will have to provide the test_dataset and at the end of the evaluation a .csv file file will be generated containing the simplified sentences predicted by the model and the evaluation metrics.

## Structure

## **src** 
Inside the src folder you can find two subfolders named: **data** and **model**.  A `settings.py` file is available: whithin it all local paths are assigned to global variables, which are reused in the scripts to access data.

### **data** folder

#### data merging
This folder is quite composite:

- the two scripts `parse_TT.py` and `parse_SIM.py` used to parse both the folders in which the dataset sentences (Terence and Teacher) were stored and the .txt file in which the original and simplified Simpitiki sentences were contained;

- the script `open_paccsit.py` where PACCSS-IT is parsed and the parallel sentences are collected;

- the script `join_datasets.py` allows you to merge the three individual datasets into a final one;

- the two scripts `clean_text.py` and `preprocess_paccsit.py` are used for pre-processing analyses to clean up datasets, either lemmatising, removing stopwords, or filtering for cosine similarity between complex and simple sentences. Various tests and trials were conducted during the project.

#### data processing 
Inside the **analysis** folder there are scripts for doing quantitative analyses and making qualitative judgements (as far as possible) on the datasets used. These are simple *NLP* analyses and the main library used is *spacy*. By running `main.py`, one can obtain graphs and metrics, including the distribution of Part Of Speech, see how many simple structures (SVO - Subject Verb Object -) are present in both simple and complex sentences, and understand the Mood and Tense of verbs.

#### data augmentation

### **model** folder
Inside this folder you will find another one called deep_martin_final which is structured like that:

- ***source***:

  - **preprocessing**: this folder contains the file `dataset.py`, which allows you to create an instance of the HuggingFaceDataset class;
  
  - **training**: this folder include the `hf_training.py` script which sets the complete training pipeline and the `hyperparameter_search.py` script which     is in charge of finding the best hyperparameters in the searching space;
  
  - **evaluation**: this folder include the `evaluator.py` script which sets the complete evaluation pipeline;
  
  - `hf_transformer_trainer.py`: is the script which actually launch the training.


All the input datasets, the pre-processed ones and those generated in the Hugging Face format needed to tune the model, are not published, partly because of the large size. The online sources from which I primarily retrieved the three corpus used in this project are as follows: 

- [PACCSS-IT (Parallel Corpus of Complex-Simple Sentences for ITalian)](http://www.italianlp.it/resources/paccss-it-parallel-corpus-of-complex-simple-sentences-for-italian/)

- [TERENCE and TEACHER](http://www.italianlp.it/resources/terence-and-teacher/)

- [SIMPITIKI](https://github.com/dhfbk/simpitiki)


### License
