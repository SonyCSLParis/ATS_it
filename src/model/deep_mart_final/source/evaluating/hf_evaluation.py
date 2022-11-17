#from src.preprocessing.dataset import HuggingFaceDataset
import pandas as pd
import numpy as np
import evaluate
import torch
from datasets import load_metric
import wandb
from src.model.deep_mart_final.source.analysis.evaluator import CTransformerEvaluator
from src.model.deep_mart_final.source.preprocessing.dataset import HuggingFaceDataset
from transformers import (
    AutoModel,
    BertTokenizer,
    AutoTokenizer,
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
                                            remove_columns_list= [],
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
    "no_repeat_ngram_size": 0,
    "length_penalty": 1.0,
    "num_beams": 1,
}


model = setup_model(model_config=model_config_dict,
                    model_path=None,
                    pretrained_model_path='/Users/francesca/Desktop/deep_mart_final/model_deep/trained_model/hf_6_epochs/checkpoint-27000',
                    resume=True,
                    tie_encoder_decoder=False,
                    tokenizer=tokenizer)

#load dataset (full)
dataset = load_dataset('/Users/francesca/Desktop/deep_mart_final/data/df_test_ultimated.csv', tokenizer_id=tokenizer_id)
#split the dataset
dataset1 = dataset.train_test_split(shuffle=False, test_size=0.10)
#obtain train and test split
train_ds = dataset1["train"].shuffle(seed=42)
test_ds = dataset1["test"]

#dataset1["validation"] = dataset1.pop("test")

#create the data collator (it will be useful to create the batches of reference sentences for later testing)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch = data_collator([train_ds[i] for i in range(1, 3)])

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


#funzione che permette di generare il bleu score a partire da un sampling casuale di frasi tratte dal nostro test_set
#che viene fornito in input
def compute_metrics(dataset):
    all_preds = []
    all_labels = []
    sampled_dataset = dataset["test"].shuffle().select(range(200))
    #questa funzione to_tf_dataset di tensorflow non eè riconosciuta ed è quella che ci genera errore
    tf_generate_dataset = sampled_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=4,
    )

    for batch in tf_generate_dataset:
        #generare predizioni
        predictions = model.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        #decodificare le predizioni
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #ricavare le labels
        labels = batch["labels"].numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)

    result = metric.compute(predictions=all_preds, references=all_labels)
    return {"bleu": result["score"]}

#first attempt of generating evaluation metrics of a test batch
print(compute_metrics(dataset1))

'''
#questa funzione è praticamente identica a quella di prima, solo che riceve in input le predizioni già compiute
def compute_metrics2(eval_preds):
    preds, labels = eval_preds
    # In case the model_deep returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


#TOWARDS DATA SCIENCE Chris Lemke
#our model_deep is using pre-trained BERT weights

#ho commentato questa parte perchè avevamo già fatto il setting sopra, ma è sempre possibile utilizzarla in seguito

model_deep = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('/Users/francesca/Desktop/deep-martin-final/model_deep/pretrained_english')
model_deep.save_pretrained('/Users/francesca/Desktop/deep-martin-final/model_deep/pretrained_english')

model_deep.config.decoder_start_token_id = tokenizer.cls_token_id
model_deep.config.eos_token_id = tokenizer.sep_token_id
model_deep.config.pad_token_id = tokenizer.pad_token_id
model_deep.config.vocab_size = model_deep.config.encoder.vocab_size

model_deep.config.max_length = 120
model_deep.config.min_length = 40
model_deep.config.early_stopping = True
model_deep.config.length_penalty = 0.8
model_deep.config.num_beams = 3


dataset = load_dataset('/Users/francesca/Desktop/deep-martin-final/data/final_hf.csv', tokenizer_id='dbmdz/bert-base-italian-xxl-cased')
dataset1 = dataset.train_test_split(shuffle=False, test_size=0.10)
train_ds = dataset1["train"].shuffle(seed=42)
test_ds = dataset1["test"]



meteor = load_metric('meteor')
rouge = load_metric('rouge')

def compute_metrics(prediction):
    labels_ids = prediction.label_ids
    pred_ids = prediction.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    meteor_output = meteor.compute(predictions=pred_str, references=label_str)
    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=['rouge2'])['rouge2'].mid

    return {
        'meteor_score': round(meteor_output['meteor'], 4),
        'rouge2_precision': round(rouge_output.precision, 4),
        'rouge2_recall': round(rouge_output.recall, 4),
        'rouge2_f_measure': round(rouge_output.fmeasure, 4)
    }

training_arguments = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy='steps',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=torch.cuda.is_available(),
    output_dir='/Users/francesca/Desktop/deep-martin-final/model_deep/trained_model/model_tutorial',
    logging_steps=100,
    save_steps=2000,
    eval_steps=10000,
    warmup_steps=2000,
    gradient_accumulation_steps=1,
    save_total_limit=3
)

trainer = Seq2SeqTrainer(
    model_deep=model_deep,
    tokenizer=tokenizer,
    args=training_arguments,
    #compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

#qui inizia il test
wandb.init(mode="disabled")
trainer.train()
trainer.save_model('/Users/francesca/Desktop/deep-martin-master/model_deep/trained_model/model_tutorial')



#input_text = '''
#Questa ragazza è estremamente simpatica
'''

trained_model = EncoderDecoderModel.from_pretrained('/Users/francesca/Desktop/deep-martin-master/model_deep/trained_model/checkpoint-3000')
tokenizer = AutoTokenizer.from_pretrained('/Users/francesca/Desktop/deep-martin-master/model_deep/trained_model/checkpoint-3000')

inputs = tokenizer([input_text], padding='max_length',
                       max_length=20, truncation=True, return_tensors='pt')

trained_model.config.decoder_start_token_id = tokenizer.cls_token_id
trained_model.config.eos_token_id = tokenizer.sep_token_id
trained_model.config.pad_token_id = tokenizer.pad_token_id
trained_model.config.vocab_size = model_deep.config.encoder.vocab_size

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
'''
