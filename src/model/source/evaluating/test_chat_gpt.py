from typing import Dict, List, Optional, Tuple
import pandas as pd
import evaluate
from src.model.source.evaluating.easse_sari.sari import corpus_sari

filename = 'add the directory where you stored test_chat_gpt.csv file'
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

    sum_sari_easse_chat = 0
    sum_sari_easse_adap = 0
    sum_sari_hf_chat = 0
    sum_sari_hf_adap = 0
    sum_bleu_chat = 0
    sum_bleu_adap = 0

    with open('/Users/martina/Desktop/test_chat_gpt.csv', 'r', errors='ignore') as file:
        df = pd.read_csv(file, sep=';', error_bad_lines=False)
        diff_list = []

        i = 0
        for row in df.iterrows():
            source_1 = row[1][1]
            source_2 = row[1][2]
            chat_simpl = row[1][3]
            reference = row[1][1]
            out_adaptive_model = row[1][4]


            '''print('COMPLEX:     ',source)
            print('REFERENCE:      ', reference)
            print('SIMPLIFICATION CHAT GPT:     ', chat_simpl)
            print('SIMPLIFICATION ADAPTIVE MODEL:     ', out_adaptive_model)'''

            sari_easse_chat = corpus_sari(orig_sents=[source_2],
                                      sys_sents=[chat_simpl],
                                      refs_sents=[[reference]])

            sari_easse_adap = corpus_sari(orig_sents=[source_1],
                                          sys_sents=[out_adaptive_model],
                                          refs_sents=[[reference]])

            bleu_chat = eval_blue_score([chat_simpl], [[reference]])['score']
            bleu_adap = eval_blue_score([out_adaptive_model], [[reference]])['score']


            sari_hf_chat = eval_sari_score(sources=[source_2], predictions=[chat_simpl], references=[[reference]])['sari_score']
            sari_hf_adap = eval_sari_score(sources=[source_1], predictions=[out_adaptive_model], references=[[reference]])['sari_score']


            sum_sari_easse_chat += sari_easse_chat
            sum_sari_easse_adap += sari_easse_adap
            sum_bleu_chat += bleu_chat
            sum_bleu_adap += bleu_adap
            sum_sari_hf_chat += sari_hf_chat
            sum_sari_hf_adap += sari_hf_adap

            i += 1

            diff = abs(sari_hf_chat - sari_hf_adap)
            diff_list.append((diff, source_2, chat_simpl, source_1, out_adaptive_model))

    #print('bleu_hf ',sum_blue//i)
    print('SARI (Easse package) for ChatGPT simplifications ', sum_sari_easse_chat/i)
    print('SARI (Easse package) for our Adaptive Model simplifications ', sum_sari_easse_adap / i)
    print('SARI (Hugging Face) for ChatGPT simplifications ', sum_sari_hf_chat / i)
    print('SARI (Hugging Face) for our Adaptive Model simplifications ', sum_sari_hf_adap / i)
    print('BLEU (Hugging Face) for ChatGPT simplifications ', sum_bleu_chat / i)
    print('BLEU (Hugging Face) for our Adaptive Model simplifications ', sum_bleu_adap / i)

    diff_list.sort(reverse=True)

    top_10_highest_diff = pd.DataFrame(diff_list[:10],
                                       columns=['Difference', 'Source 2', 'ChatGPT Simplification', 'Source 1',
                                                'Adaptive Model Simplification'])

    top_10_highest_diff.to_csv('/Users/martina/Desktop/top_10_highest_differences.csv', index=False)

    for i, (diff, source_2, chat_simpl, source_1, out_adaptive_model) in enumerate(diff_list[:10]):
        print(f'Difference {i + 1}: {diff}')
        print(f'Source 2: {source_2}')
        print(f'ChatGPT Simplification: {chat_simpl}')
        print(f'Source 1: {source_1}')
        print(f'Adaptive Model Simplification: {out_adaptive_model}')

    diff_list.sort(reverse=False)

    top_10_lowest_diff = pd.DataFrame(diff_list[:10],
                                      columns=['Difference', 'Source 2', 'ChatGPT Simplification', 'Source 1',
                                               'Adaptive Model Simplification'])
    top_10_lowest_diff.to_csv('/Users/martina/Desktop/top_10_lowest_differences.csv', index=False)

    for i, (diff, source_2, chat_simpl, source_1, out_adaptive_model) in enumerate(diff_list[:10]):
        print(f'Difference {i + 1}: {diff}')
        print(f'Source 2: {source_2}')
        print(f'ChatGPT Simplification: {chat_simpl}')
        print(f'Source 1: {source_1}')
        print(f'Adaptive Model Simplification: {out_adaptive_model}')


except pd.errors.ParserError as e:
    print('Error occurred while parsing the CSV file:', str(e))
