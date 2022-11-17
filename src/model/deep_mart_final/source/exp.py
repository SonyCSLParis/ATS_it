'''from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import functools
import gc
import logging
import os
import sys
from typing import Dict, Optional, Tuple
from src.preprocessing.dataset import HuggingFaceDataset
import torch
import wandb
import pandas as pd'''


'''documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model_deep = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
fname = get_tmpfile("my_doc2vec_model")

model_deep.save(fname)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model_deep = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
texts = ["Hello there!", "This is another text"]
tokenized_texts = tokenizer(texts, padding=True)


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


test_dataset = SimpleDataset(tokenized_texts)
trainer = Trainer(model_deep=model_deep)
predictions = trainer.predict(test_dataset)
print(predictions.predictions,
      predictions.label_ids)


model_deep.eval()
pt_inputs = {k: torch.tensor(v).to(trainer.args.device) for k, v in tokenized_texts.items()}
with torch.no_grad():
    output = model_deep(**pt_inputs)
print(output.logits.cpu().numpy())



from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("philippelaban/keep_it_simple")
kis_model = AutoModelForCausalLM.from_pretrained("philippelaban/keep_it_simple")

paragraph = """A small capsule containing asteroid soil samples that was dropped from 136,700 miles in space by Japan's Hayabusa2 spacecraft landed as planned in the Australian Outback on December 6. The extremely high precision required to carry out the mission thrilled many in Japan, who said they took pride in its success."""

start_id = tokenizer.bos_token_id
tokenized_paragraph = [(tokenizer.encode(text=paragraph) + [start_id])]
input_ids = torch.LongTensor(tokenized_paragraph)

output_ids = kis_model.generate(input_ids, max_length=150, num_beams=4, do_sample=True, num_return_sequences=8)
output_ids = output_ids[:, input_ids.shape[1]:]
output = tokenizer.batch_decode(output_ids)
output = [o.replace(tokenizer.eos_token, "") for o in output]

for o in output:
    print("----")
    print(o)'''
import pandas as pd
from datasets import Dataset, load_metric
import datasets
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from src.PACCS.settings import *
from src.model.deep_mart_final.settings2 import *
dummy_bert2bert = EncoderDecoderModel.from_pretrained( TRAINED_MODELS_DIR+ "/checkpoints/checkpoint-20")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
data_path = OUTPUT_DIR + '/df_test_ultimated.csv'

df2 = pd.read_csv(data_path)
colonna_complessa = [str(riga) for riga in list(df2['Normal'])]
colonna_semplice = [str(riga) for riga in list(df2['Simple'])]

datafr = pd.DataFrame({"Normal": colonna_complessa, "Simple": colonna_semplice})
test_data = Dataset.from_pandas(datafr)
dataset1 = test_data.train_test_split(shuffle=True, test_size=0.10)

test_ds2 = dataset1['train'].shuffle(seed=42)
eval_ds = dataset1['test']


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


    meteor_output =meteor.compute(
        predictions=pred_str, references=label_str
    )

    return {
            "meteor_score": round(meteor_output["meteor"], 4),
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

def generate_summary(batch):
    # cut off at BERT max length 512
    inputs = tokenizer(batch["Normal"], padding="max_length", truncation=True, max_length=20, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = dummy_bert2bert.generate(input_ids, attention_mask=attention_mask, max_new_tokens = 20)
    print(type(outputs))
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(type(output_str))
    print(output_str)
    batch["pred_simplification"] = output_str

    return batch

batch_size = 64  # change to 64 for full evaluation

results = test_ds2.map(generate_summary,
                        batched=True,
                        batch_size=batch_size,
                        remove_columns=["Normal"])

print(results["pred_simplification"])

print(rouge.compute(predictions=results["pred_simplification"], references=results["Simple"], rouge_types=["rouge2"])["rouge2"].mid)
