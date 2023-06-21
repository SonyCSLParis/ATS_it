import logging
import sys
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from datasets import load_metric
from tqdm import tqdm
from transformers import EncoderDecoderModel, AutoTokenizer
from settings import *
import evaluate
import sacrebleu
from src.model.source.evaluating.easse_sari.sari import corpus_sari
import csv


def eval_blue_score(predictions, references):
    blue = evaluate.load("sacrebleu")
    results = blue.compute(predictions=predictions, references=references)
    return {
        'score': round(results["score"], 1),
        'precision': results["precisions"]}

def eval_sari_score(
        sources: List[str], predictions: List[str], references: List[List[str]]
) -> Dict:
    sari = evaluate.load("sari")
    result = sari.compute(
        sources=sources, predictions=predictions, references=references
    )
    return {"sari_score": round(result["sari"], 4)}

try:

    sum_sari = 0
    sum_blue = 0
    with open('/Users/utente/Desktop/test_chat_gpt.csv', 'r', errors='ignore') as file:
        df = pd.read_csv(file, sep=';', error_bad_lines=False)
        i = 0
        for row in df.iterrows():
            source = row[1][2]
            out = row[1][3]
            references = row[1][1]
            print('COMPLEX:     ',row[1][2])
            print('REFERENCE:      ', row[1][1])
            print('SIMPLIFICATION:     ', row[1][3])

            sari_easse1 = corpus_sari(orig_sents=[source],
                                      sys_sents=[out],
                                      refs_sents=[[references]])
            #bleu = eval_blue_score([out], [[references]])['score']
            sari = eval_sari_score(sources=[source], predictions=[out], references=[[references]])['sari_score']
            #print('bleu = ', bleu)
            #print('sari_easse = ', sari_easse1)
            #print('sari_metric = ', sari)

            sum_sari += sari
            i += 1
    print(sum_sari//i)
    #print(sum_blue//i)



except pd.errors.ParserError as e:
    print('Error occurred while parsing the CSV file:', str(e))