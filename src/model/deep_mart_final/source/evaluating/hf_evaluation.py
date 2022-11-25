#from src.preprocessing.dataset import HuggingFaceDataset
import tensorflow as tf
import pandas as pd
import numpy as np
import evaluate
import torch
from datasets import load_metric
import wandb

from src.model.deep_mart_final.source.preprocessing.dataset import HuggingFaceDataset
from transformers import (
    AutoModel,
    BertTokenizer,
    AutoTokenizer,
    AutoConfig,
    EncoderDecoderModel,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


tokenizer_id = 'dbmdz/bert-base-italian-xxl-cased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

#Qui ci siamo semplicemnte salvati il modello e il tokenizer (pre-trained on Bert) nella nostra cartella dei modelli pre-training
# model_deep = EncoderDecoderModel.from_encoder_decoder_pretrained(tokenizer_id, tokenizer_id)
# tokenizer.save_pretrained('/Users/francesca/Desktop/deep-martin-master/model_deep/pretrained_italian')
# model_deep.save_pretrained('/Users/francesca/Desktop/deep-martin-master/model_deep/pretrained_italian')

#abbiamo importato la funzione che ci permette di fare il loading del nostro dataset
def load_dataset(path, tokenizer_id):
    df = pd.read_csv(path)
    colonna_complessa = [str(riga) for riga in list(df['Sentence_1'])]
    colonna_semplice = [str(riga) for riga in list(df['Sentence_2'])]

    dataframe = pd.DataFrame({"Normal": colonna_complessa, "Simple": colonna_semplice})

    dataset = HuggingFaceDataset.hf_dataset(dataframe,
                                            remove_columns_list= ['Normal', 'Simple'],
                                            identifier = tokenizer_id,
                                            batch_size=8)

    return dataset


#qui sotto inizializziamo correttamente il modello pre-trained
def setup_model(
        model_config,
        model_path,
        pretrained_model_path,
        resume,
        tie_encoder_decoder,
        tokenizer,
):
    if resume:
        model = EncoderDecoderModel.from_pretrained(pretrained_model_path)

    elif pretrained_model_path is not None and model_path is None:
        model = EncoderDecoderModel.from_pretrained(pretrained_model_path)


    elif pretrained_model_path is not None:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            model_path, model_path, tie_encoder_decoder=tie_encoder_decoder
        )
        model.save_pretrained(pretrained_model_path)

    else:
        raise ValueError(
            "Please provide either `pretrained_model_path` or `model_path` and `pretrained_model_path`."
        )

    if tokenizer.name_or_path != "facebook/bart-base":
        model.config.vocab_size = model.config.encoder.vocab_size


    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = model_config["max_length"]
    model.config.min_length = model_config["min_length"]
    model.config.no_repeat_ngram_size = model_config["no_repeat_ngram_size"]
    model.config.early_stopping = True
    model.config.length_penalty = model_config["length_penalty"]
    model.config.num_beams = model_config["num_beams"]
    return model

#qui creiamo un dizionario con le configurazioni importanti del nostro modello
model_config_dict = {
    "max_length": 20,
    "min_length": 0,
    "no_repeat_ngram_size": 3,
    "length_penalty": -0.5,
    "num_beams": 1,
}


modello_1= setup_model(model_config=model_config_dict,
                    model_path=None,
                    pretrained_model_path='/Users/francesca/Desktop/Github/Final/src/model/deep_mart_final/model_deep/trained_model/checkpoint-4000',
                    resume=True,
                    tie_encoder_decoder=False,
                    tokenizer=tokenizer)





#load dataset (full)
dataset = load_dataset('/Users/francesca/Desktop/Github/Final/output/ultimated.csv', tokenizer_id=tokenizer_id)
#split the dataset
dataset1 = dataset.train_test_split(shuffle=False, test_size=0.10)
#obtain train and test split
train_ds = dataset1["train"].shuffle(seed=42)
test_ds = dataset1["test"]



#dataset1["validation"] = dataset1.pop("test")

#create the data collator (it will be useful to create the batches of reference sentences for later testing)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=modello_1)
batch = data_collator([train_ds[i] for i in range(1, 5)])

#download the metric of interest
metric = evaluate.load("sacrebleu")

#this is just a dummy example
predictions = ["Questa è la vita"]
references = [
    [
        'Questa è forse la vita'
    ]
]
print(metric.compute(predictions=predictions, references=references))


frase = 'Marta ha sporcato i calzini di salsa'
inputs = tokenizer([frase], padding='max_length', max_length=20, truncation=True, return_tensors='pt')
predictions = modello_1.generate(inputs['input_ids'], max_length=20, min_length=0,
                                 num_beams=1,
                                 length_penalty=1.0,
                                 # emperature=1.0,
                                 early_stopping=True,
                                 top_k=50,
                                 do_sample=False)

#funzione che permette di generare il bleu score a partire da un sampling casuale di frasi tratte dal nostro test_set
#che viene fornito in input
def compute_metrics(dataset, modello):
    all_preds = []
    all_labels = []
    sampled_dataset = dataset["test"].shuffle().select(range(200))
    tf_generate_dataset = sampled_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=4,
    )
    #questa funzione to_tf_dataset di tensorflow non eè riconosciuta ed è quella che ci genera errore

    for batch in tf_generate_dataset:

        predictions = modello.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        print(predictions)
        labels_ids = predictions.label_ids
        pred_ids = predictions.predictions


        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in pred_str]
        decoded_labels = [[label.strip()] for label in label_str]
        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)

    #first attempt of generating evaluation metrics of a test batch

    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=all_preds, references=all_labels)
    return {"bleu": result["score"]}



print(compute_metrics(dataset1, modello_1))


input_text = '''
Questa ragazza è estremamente simpatica
'''

trained_model = EncoderDecoderModel.from_pretrained('/Users/francesca/Desktop/Github/Final/src/model/deep_mart_final/model_deep/trained_model/checkpoint-4000')
tokenizer = AutoTokenizer.from_pretrained('/Users/francesca/Desktop/Github/Final/src/model/deep_mart_final/model_deep/trained_model/checkpoint-4000')

inputs = tokenizer([input_text], padding='max_length',
                       max_length=20, truncation=True, return_tensors='pt')

trained_model.config.decoder_start_token_id = tokenizer.cls_token_id
trained_model.config.eos_token_id = tokenizer.sep_token_id
trained_model.config.pad_token_id = tokenizer.pad_token_id
trained_model.config.vocab_size = trained_model.config.encoder.vocab_size

output = trained_model.generate(inputs['input_ids'],
                        max_length=20,
                        min_length=0,
                        num_beams=1,
                        length_penalty=1.0,
                        #temperature=1.0,
                        #early_stopping=True,
                        #top_k=50,
                        do_sample=False)

#qui mi restituisce errore, perchè?
#labels_ids = output.label_ids (a quanto pare è proprio legato alla questione tensorflow)
#pred_ids = output.predictions

#pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#labels_ids[labels_ids == -100] = tokenizer.pad_token_id
#text1 = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
print(output)
text = tokenizer.batch_decode(output, skip_special_tokens=True)
print(text)

