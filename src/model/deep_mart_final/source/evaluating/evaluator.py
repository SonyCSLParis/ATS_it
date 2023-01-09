import logging
import sys
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from datasets import load_metric
from tqdm import tqdm
from transformers import EncoderDecoderModel, AutoTokenizer
from src.settings import *
import evaluate

'''Note: Parts of this code are lifted as is from those written by Christopher Lemcke.

Copyright (c) 2022, Cristopher Lemcke <github: https://github.com/chrislemke/deep-martin
'''

class HFEvaluator:
    '''
    This class deals with the entire performance evaluation part of the model.
    '''
    def __init__(
            self,
            eval_dataset_path: str,
            model_path: str,
            tokenizer_path: Optional[str] = None,
            log_level="WARNING"):

        '''
        The class is initialized with some parameters.
        :param eval_dataset_path: The path to the dataset on which is made teh evaluation.
        :param model_path: The path of the model which will be tested and evaluated.
        :param tokenizer_path: The path to the tokenizer which we are going to use.
        :param log_level: which is the level for the logging of teh parameters
        '''

        #open the evaluation dataset
        df = pd.read_csv(eval_dataset_path, index_col=False)
        self.df = df
        self.logger = logging.getLogger(__name__)

        #loading of all the metrics that will be used in order to conduct the evaluation
        self.blue = evaluate.load("sacrebleu")
        self.sari = load_metric("sari")
        self.bert_score = load_metric("bertscore")
        self.rouge = load_metric("rouge")
        self.glue = load_metric("glue", "stsb")
        self.meteor = load_metric("meteor")

        #you initialize your tokenizer
        if tokenizer_path is None:
            tokenizer_path = model_path
            self.logger.info(
                f"No `tokenizer_path` provided. {model_path} will be used!"
            )
        if tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        else:
            raise ValueError(
                f"Could not find a suitable tokenizer for: {tokenizer_path}!"
            )


        # load the model trained and instantiate it
        self.model = EncoderDecoderModel.from_pretrained(model_path)

        # set up the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #logging the configuration
        logging.basicConfig(
            level=log_level,
            handlers=[logging.StreamHandler(sys.stdout)],
            format="%(levelname)s - %(message)s",
        )


    def __config_model(self, model_config: Dict):
        '''
        Here you configure the model setting by giving a dictionary as input to the function
        '''
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.encoder.vocab_size
        self.model.config.max_length = model_config["max_length"]
        self.model.config.min_length = model_config["min_length"]

        #if set to int > 0, all ngrams of that size can only occur once
        self.model.config.no_repeat_ngram_size = model_config['no_repeat_ngram_size']
        self.model.config.early_stopping = model_config['decoder']["early_stopping"]

        #exponential penalty to the length that is used with beam-based generation.
        # It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
        # Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
        self.model.config.length_penalty = model_config['length_penalty']

        #number of beams for beam search. 1 means no beam search
        self.model.config.num_beams = model_config["num_beams"]

        #the value used to module the next token probabilities
        self.model.config.temperature = model_config['decoder']["temperature"]

        #the number of highest probability vocabulary tokens to keep for top-k-filtering
        self.model.config.top_k = model_config['decoder']["top_k"]

        #if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation
        self.model.config.top_p = model_config['decoder']["top_p"]

        self.model.config.num_beam_groups = model_config['decoder']["num_beam_groups"]

        #to use sampling or not (otherwise greedy search)
        self.model.config.do_sample = model_config['decoder']["do_sample"]

        #the parameter for repetition penalty. 1.0 means no penalty
        self.model.config.repetition_penalty = model_config['decoder']["repetition_penalty"]



    def __sources_and_references(self) -> Dict:
        '''
        This function allows to iterate through all the rows of the dataset and obtain the Normal and Simplified sentences,
        it then creates a dictionaries of key/values with them.
        '''
        dictionary = {}
        for index, row in self.df.iterrows():
            key = str(row["Normal"]).replace("\n", "")
            value = str(row["Simple"]).replace("\n", "")
            dictionary[key] = value

        return dictionary



    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            model_config: Dict,
    ) -> List[str]:
        '''
        This fuction allows the model to generate new simplified sentences,
        '''


        self.__config_model(model_config)
        model = self.model.to(self.device)

        model_output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens = 50, renormalize_logits = True)

        return model_output, self.tokenizer.batch_decode(model_output, skip_special_tokens=True)



    def __tokenize(self, sources: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This function permits to tokenize correctly the sentences which receives in input.
        '''
        inputs = self.tokenizer(
            sources, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        return input_ids, attention_mask


    #here below we have the list of functions which load the metrics and return the score
    #and results according to each specific implementation
    def eval_sari_score(
            self, sources: List[str], predictions: List[str], references: List[List[str]]
    ) -> Dict:
        result = self.sari.compute(
            sources=sources, predictions=predictions, references=references
        )
        return {"sari_score": round(result["sari"], 4)}



    def eval_bert_score(self, predictions, references) -> Dict:
        result = self.bert_score.compute(
            predictions=predictions, references=references, lang="en"
        )
        return {"bert_score_f1": round(result["f1"][0], 4)}



    def eval_meteor_score(self, predictions, references) -> Dict:
        result = self.meteor.compute(predictions=predictions, references=references)
        return {"meteor_score": round(result["meteor"], 4)}



    def eval_rouge_scores(self, predictions, references ) -> Dict:
        result = self.rouge.compute(
            predictions=predictions, references=references, rouge_types=["rouge2"]
        )["rouge2"].mid
        return {
            "rouge2_precision": round(result.precision, 4),
            "rouge2_recall": round(result.recall, 4),
            "rouge2_f_measure": round(result.fmeasure, 4),
        }


    def eval_blue_score(self, predictions, references):
        results = self.blue.compute(predictions = predictions, references = references)
        return {
            'score': round(results["score"], 1),
            'precision': results["precisions"]
        }


    def eval_glue_score(self, predictions, references) -> Dict:
        result = self.glue.compute(predictions=predictions, references=references)
        return {
            "glue_pearson": round(result["pearson"], 4),
            "glue_spearman_r": round(result["spearmanr"], 4),
        }



    # this is the most important function which connects all the steps have been defined above
    def evaluate_with_dataset(
            self,
            # configuration of the model_deep
            model_config: Dict,

            # the path where to place the output
            csv_output_path: Optional[str] = None,

            extend_dataframe: bool = False,
    ):

        # creation of the output .csv file which will contain the ground truth sentences, the predictions and all the scores of the different metrics
        result_df = pd.DataFrame(columns=["Normal", "Simple", "SARI", "METEOR", "ROUGE_F",'BLEU', "SPEARMAN_CORRELATION", "PEARSON_CORRELATION"])

        # iterate through all the instances of the dictionary created by the __source_and_reference() function
        for source, references in tqdm(self.__sources_and_references().items()):

            print(source)
            inputs = self.__tokenize(source)
            print(inputs)
            reference_tokens = self.__tokenize(references)
            '''print(references)
            print(reference_tokens)'''

            undecoded_out, output = self.generate(*inputs, model_config=model_config)

            '''print(undecoded_out)
            print(output)'''



            rouge_result = self.eval_rouge_scores(
                predictions=[undecoded_out], references=[reference_tokens]
            )

            blue_result = self.eval_blue_score(predictions= output, references= [[references]])

            sari_result = self.eval_sari_score(
                sources=[source], predictions=output, references=[[references]]
            )
            meteor_result = self.eval_meteor_score(
                predictions=output, references=[references]
            )

            result_df = result_df.append(
                {
                    "Normal": source,
                    "Simple": output[0],
                    "SARI": sari_result["sari_score"],
                    "METEOR": meteor_result["meteor_score"],
                    "ROUGE_F": rouge_result['rouge2_f_measure'],
                    'BLEU': blue_result['score'],
                    "SPEARMAN_CORRELATION": 0,
                    "PEARSON_CORRELATION": 0,

                },
                ignore_index=True,
            )

        if csv_output_path is not None:
            result_df.to_csv(csv_output_path, index=False)
            print(f"Dataframe saved at: {csv_output_path}.")



# I instantiate the class, giving all the required arguments
classe = HFEvaluator(eval_dataset_path =  '/Users/francesca/Desktop/Github/Final_final/output/csv_files/augmented/test.csv',
                     model_path= '/Users/francesca/Desktop/trained_models/augmented_20',
                     tokenizer_path= '/Users/francesca/Desktop/trained_models/augmented_20',
                     log_level="WARNING")

# I first open the configuration file and upload as a dictionary, but pay attention because you have to take care of selecting correctly the elements afterwards
with open( '/Users/francesca/Desktop/trained_models/augmented_20/config.json') as json_file:
    data = json.load(json_file)
    print(data)

# I ask to evaluate the generated data
classe.evaluate_with_dataset(model_config=data,
                             csv_output_path= CSV_EVAL_OUTPUT + '/augmented_20.csv',
                             extend_dataframe=False)
