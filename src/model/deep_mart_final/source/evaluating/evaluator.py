import logging
import sys
import json
from typing import Dict, List, Optional, Tuple
from src.model.deep_mart_final.source.custom_transformer.model_implementation.transformer import get_model
import pandas as pd
import torch
from datasets import load_metric
from tqdm import tqdm
from transformers import BertTokenizerFast, EncoderDecoderModel, RobertaTokenizerFast, AutoTokenizer
from settings import *

# this class allows to create the Evaluator for the Customized Transformer that has been implemented by Christophen Lemke
class CTransformerEvaluator:

    def __init__(
            # here all the necessary components are initialized
            self,
            arguments,
            vocab_size,
            eval_dataset_path: str,  # path of the evaluation dataset
            model_path: str,  # path of the model_deep which has been trained and will perform the prediction
            #doc2vec_model_path: str,  # path for a DocToVec model_deep already trained (but how and on which documents?
            tokenizer_path: str,  # path of the tokenizer
            log_level="WARNING"):

        # reading the dataset
        self.df = pd.read_csv(eval_dataset_path, index_col=0)
        self.logger = logging.getLogger(__name__)

        # here we apply this function to the model_deep (go to the analysis.py file to understand better
        #self.analysis_helper = AnalysisHelper(doc2vec_model_path)

        # just load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # set up the machine you will use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set up and load the model_deep you have already trained

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(arguments, vocab_size, vocab_size, device)
        self.model.load_state_dict(torch.load(model_path))


        logging.basicConfig(
            level=log_level,
            handlers=[logging.StreamHandler(sys.stdout)],
            format="%(levelname)s - %(message)s",
        )



    def evaluate(self):
        self.model.eval()



class HFEvaluator:

    def __init__(
            self,
            eval_dataset_path: str,
            model_path: str,
            #doc2vec_model_path: str,
            tokenizer_path: Optional[str] = None,
            log_level="WARNING"):

        # read the csv
        df = pd.read_csv(eval_dataset_path, index_col=False)
        self.df = df.head(200)
        # you load all the metrics of interest and
        # self.logger = logging.getLogger(_name_)
        self.sari = load_metric("sari")
        self.bert_score = load_metric("bertscore")
        self.rouge = load_metric("rouge")
        self.glue = load_metric("glue", "stsb")
        self.meteor = load_metric("meteor")

        # you call that function, implemented in analysis.py file, to the doc2vec model_deep, but very uncertain on that
        #self.analysis_helper = AnalysisHelper(doc2vec_model_path)

        if tokenizer_path is None:
            tokenizer_path = model_path
            self.logger.info(
                f"No `tokenizer_path` provided. {model_path} will be used!"
            )
        if tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


        if "roberta" in tokenizer_path:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)

        elif "bert" in tokenizer_path:
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

        else:
            raise ValueError(
                f"Could not find a suitable tokenizer for: {tokenizer_path}!"
            )

        # load the model_deep
        self.model = EncoderDecoderModel.from_pretrained(model_path)

        # set up the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.basicConfig(
            level=log_level,
            handlers=[logging.StreamHandler(sys.stdout)],
            format="%(levelname)s - %(message)s",
        )

    # here you configure the model_deep setting by giving a dictionary to the function, which is saved in the same directory of the trained model_deep
    def __config_model(self, model_config: Dict):
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.encoder.vocab_size
        self.model.config.max_length = model_config['decoder']["max_length"]
        self.model.config.min_length = model_config['decoder']["min_length"]
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.early_stopping = model_config['decoder']["early_stopping"]
        self.model.config.length_penalty = model_config['decoder']["length_penalty"]
        self.model.config.num_beams = model_config['decoder']["num_beams"]
        self.model.config.temperature = model_config['decoder']["temperature"]
        self.model.config.top_k = model_config['decoder']["top_k"]
        self.model.config.top_p = model_config['decoder']["top_p"]
        self.model.config.num_beam_groups = model_config['decoder']["num_beam_groups"]
        self.model.config.do_sample = model_config['decoder']["do_sample"]

    # this function allows to iterate through all the rows of the dataset and obtain the Normal and Simplified sentences.
    # it then creates a dictionaries of key/values with them
    def __sources_and_references(self) -> Dict:
        dictionary = {}
        for index, row in self.df.iterrows():
            key = str(row["Normal"]).replace("\n", "")
            value = str(row["Simple"]).replace("\n", "")
            dictionary[key] = value

        return dictionary


    # this function is the one we found to be used before on the model_deep
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            model_config: Dict,
    ) -> List[str]:

        self.__config_model(model_config)
        model = self.model.to(self.device)
        model_output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens = 20)

        return model_output, self.tokenizer.batch_decode(model_output, skip_special_tokens=True)



    def __tokenize(self, sources: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            sources, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        return input_ids, attention_mask



    def eval_sari_score(
            self, sources: List[str], predictions: List[str], references: List[List[str]]
    ) -> Dict:
        result = self.sari.compute(
            sources=sources, predictions=predictions, references=references
        )
        return {"sari_score": round(result["sari"], 4)}



    def eval_bert_score(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.bert_score.compute(
            predictions=predictions, references=references, lang="en"
        )
        return {"bert_score_f1": round(result["f1"][0], 4)}



    def eval_meteor_score(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.meteor.compute(predictions=predictions, references=references)
        return {"meteor_score": round(result["meteor"], 4)}



    def eval_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.rouge.compute(
            predictions=predictions, references=references, rouge_types=["rouge2"]
        )["rouge2"].mid
        return {
            "rouge2_precision": round(result.precision, 4),
            "rouge2_recall": round(result.recall, 4),
            "rouge2_f_measure": round(result.fmeasure, 4),
        }



    def eval_glue_score(self, predictions: List[int], references: List[int]) -> Dict:
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

            # I don't know what does it mean
            extend_dataframe: bool = False,
    ):

        # creation of the output csv file that will be shown to us as output
        result_df = pd.DataFrame(columns=["Normal", "Simple", "SARI", "METEOR", "ROUGE_F", "SPEARMAN_CORRELATION", "PEARSON_CORRELATION"])

        # iterate through all the instances of the dictionary created by the __source_and_reference() function
        for source, references in tqdm(self.__sources_and_references().items()):

            print(source)
            inputs = self.__tokenize(source)
            print(inputs)
            reference_tokens = self.__tokenize(references)
            print(references)
            print(reference_tokens)
            undecoded_out, output = self.generate(*inputs, model_config=model_config)

            print(undecoded_out)
            print(output)

            glue_result = self.eval_glue_score(
                predictions=undecoded_out[0][0:20].tolist(), # 0:21 va cambiato quando capiamo come far generare inputs e reference da massimo 20
                references=reference_tokens[0][0].tolist(),
            )

            rouge_result = self.eval_rouge_scores(
                predictions=output, references=[references]
            )


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
                    "ROUGE_F": rouge_result["rouge2_f_measure"],
                    "SPEARMAN_CORRELATION": glue_result["glue_spearman_r"],
                    "PEARSON_CORRELATION": glue_result["glue_pearson"],
                },
                ignore_index=True,
            )

        if csv_output_path is not None:
            result_df.to_csv(csv_output_path, index=False)
            print(f"Dataframe saved at: {csv_output_path}.")



# I instantiate the class, giving all the required arguments
classe = HFEvaluator(eval_dataset_path =  OUTPUT_DIR + '/df_test_ultimated.csv',
                     model_path= '/Users/francesca/Desktop/Github/Final/src/model/deep_mart_final/model_deep/trained_model/LAST_HF_1/checkpoint-6000',
                     tokenizer_path= '/Users/francesca/Desktop/Github/Final/src/model/deep_mart_final/source/bert2bert',
                     log_level="WARNING")

# I first open the configuration file and upload as a dictionary, but pay attention because you have to take care of selecting correctly the elements afterwards
with open('/Users/francesca/Desktop/Github/Final/src/model/deep_mart_final/model_deep/trained_model/LAST_HF_1/checkpoint-6000/config.json') as json_file:
    data = json.load(json_file)

# I ask to evaluate the generated data
classe.evaluate_with_dataset(model_config=data,
                             csv_output_path=  CSV_EVAL_OUTPUT + '/evaluation3.csv',
                             extend_dataframe=False)
