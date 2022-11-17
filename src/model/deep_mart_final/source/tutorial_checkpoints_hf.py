from transformers import AutoTokenizer, EncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
import datasets
import wandb
from src.deep_martin_final.source.training.hf_training import HuggingFaceTrainer
from datasets import Dataset
wandb.login(key='REDACTED')
wandb.init(project="ATS", entity="francapado")

#path to the joined dataset
data_path = '/Users/francesca/Desktop/14_11_22/src/deep_martin_final/source/join_dataset/ultimated.csv'
#tokenizer identifier
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")

#open the dataset as a csv
df = pd.read_csv(data_path)

#process the dataset
colonna_complessa = [str(riga) for riga in list(df['Sentence_1'])]
colonna_semplice = [str(riga) for riga in list(df['Sentence_2'])]
dataframe = pd.DataFrame({"Normal": colonna_complessa, "Simple": colonna_semplice})

#open it as a full train dataset
train_data = Dataset.from_pandas(dataframe)
dataset1 = train_data.train_test_split(shuffle=True, test_size=0.10)
train_ds = dataset1["train"].shuffle(seed=42)
test_ds = dataset1["test"]

dataset2 = test_ds.train_test_split(shuffle = True, test_size = 0.40)
test_ds2 = dataset2['train'].shuffle(seed=42)
eval_ds = dataset2['test']


# map article and summary len to dict as well as if sample is longer than 50 tokens
def map_to_length(x):
  x["normal_len"] = len(tokenizer(x['Normal']))
  x["normal_longer_50"] = (len(tokenizer(x["Normal"])) > 50)
  x["simple_len"] = len(tokenizer(x["Simple"]).input_ids)
  x["simple_longer_15"] = int(x["simple_len"] > 15)
  x["simple_longer_30"] = int(x["simple_len"] > 30)
  return x

sample_size = 20000
data_stats = train_data.select(range(sample_size)).map(map_to_length, num_proc=4)

def compute_and_print_stats(x):
  if len(x["normal_len"]) == sample_size:
    print(
        "Article Mean: {}, %-Articles > 50:{}, Simple Mean:{}, %-Simple > 64:{}, %-Simple > 128:{}".format(
            sum(x["normal_len"]) / sample_size,
            sum(x["normal_longer_50"]) / sample_size,
            sum(x["simple_len"]) / sample_size,
            sum(x["simple_longer_15"]) / sample_size,
            sum(x["simple_longer_30"]) / sample_size,
        )
    )

output = data_stats.map(
  compute_and_print_stats,
  batched=True,
  batch_size=-1,
)

#Here finishes the initial inspection of the data and starts the setting of the training phase
##################################################################################

train_data = train_data.select(range(50000))


def process_data_to_model_inputs(batch):
    encoder_max_length = 80
    decoder_max_length = 20
    # tokenize the inputs and labels
    inputs = tokenizer(batch["Normal"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["Simple"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

batch_size=4

#TRAIN DATA SETTING
train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["Normal", "Simple"]
)

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)



#VALIDATION DATA SETTING
val_data = eval_ds.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["Normal", "Simple"]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

#CONFIGURE THE MODEL PARAMETERS
bert2bert = EncoderDecoderModel.from_pretrained('/Users/francesca/Desktop/14_11_22/src/deep_martin_final/source/bert2bert')
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

bert2bert.config.max_length = 20
bert2bert.config.min_length = 2
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="/Users/francesca/Desktop/14_11_22/src/deep_martin_final/model/trained_model",
    save_steps=5000,
    #eval_steps=4,
    num_train_epochs = 6,
    #logging_steps=1000,
    #save_steps=500,
    #eval_steps=7500,
    warmup_steps=100,
    save_total_limit=5,
)

bert_score = datasets.load_metric("bertscore")
rouge = datasets.load_metric("rouge")
meteor = datasets.load_metric("meteor")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    bert_score_output = bert_score.compute(
        predictions=pred_str, references=label_str, lang="en"
    )

    meteor_output =meteor.compute(
        predictions=pred_str, references=label_str
    )

    return {"bert_score_f1": round(bert_score_output["f1"][0], 4),
            "meteor_score": round(meteor_output["meteor"], 4),
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()

