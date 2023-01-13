import functools
import gc
import logging
import os
import sys
import optuna
from typing import Dict, Optional, Tuple
from src.model.source.preprocessing.dataset import HuggingFaceDataset
import torch
import wandb
from transformers import logging as hf_logging
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    EvalPrediction,
    Seq2SeqTrainer,
    AutoConfig,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

'''Note: Parts of this code are lifted as is from those written by Christopher Lemcke.

Copyright (c) 2022, Cristopher Lemcke <github: https://github.com/chrislemke/deep-martin
'''

class HuggingFaceTrainer:
    '''
    This class is the one in charge of implementing the complete pipeline for starting model training.
    It includes a whole series of methods that are required to load the data,
    process it and prepare it to give it to the model and begin fine-tuning.
    '''
    __rouge = load_metric("rouge")
    __bert_score = load_metric("bertscore")
    __meteor = load_metric("meteor")
    __logger = logging.getLogger(__name__)

    @staticmethod
    def setup_logger(level: str = "INFO"):
        '''
        Just an utility function
        '''
        logging.basicConfig(
            level=logging.getLevelName(level),
            handlers=[logging.StreamHandler(sys.stdout)],
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    @staticmethod
    def load_dataset(ds_path) -> Tuple[Dataset, Dataset]:
        '''
        The function to load the dataset in the format required to provide the model with data in the correct way
        :param ds_path: it's the path to the Dataset (in Hugging Face format) already saved on disk, local directory
        :return: the two training and test dataset already pre-processed
        '''

        path_first = ds_path + '/train'
        path_second = ds_path + '/validation'

        dataset_tr, dataset_ts = HuggingFaceDataset.hf_dataset(
            path1=path_first,
            path2=path_second,
            remove_columns_list=['Normal', 'Simple'],
            identifier="dbmdz/bert-base-italian-xxl-cased",
            batch_size=8)


        HuggingFaceTrainer.__logger.info(
            f" Loaded train_dataset length is: {len(dataset_tr)}."
        )
        HuggingFaceTrainer.__logger.info(
            f" Loaded test_dataset length is: {len(dataset_ts)}."
        )

        return dataset_tr, dataset_ts



    @staticmethod
    def compute_metrics(auto_tokenizer, prediction: EvalPrediction):
        '''
        This function is then injected in the Trainer class and allows to calculate the metrics during the eveluation phase.
        Metrics are important for evaluating a model’s predictions.
        :param auto_tokenizer: Type of tokenizer to be used
        :param prediction: it's the prediction made by the model, over an evaluation set
        :return: the value of the metrics which we have chosen for our evaluation phase
        '''
        tokenizer = auto_tokenizer

        labels_ids = prediction.label_ids
        pred_ids = prediction.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = HuggingFaceTrainer.__rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        bert_score_output = HuggingFaceTrainer.__bert_score.compute(
            predictions=pred_str, references=label_str, lang="en"
        )

        meteor_output = HuggingFaceTrainer.__meteor.compute(
            predictions=pred_str, references=label_str
        )
        return {
            "bert_score_f1": round(bert_score_output["f1"][0], 4),
            "meteor_score": round(meteor_output["meteor"], 4),
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_f_measure": round(rouge_output.fmeasure, 4),
        }



    @staticmethod
    def setup_wandb(resume, training_config, wandb_config):
        '''
        We set up the wandb environment.
        '''
        if wandb_config is not None:
            HuggingFaceTrainer.__logger.info(
                f"Starting Wandb with:\n"
                f"API-key: {wandb_config['api_key']}\n"
                f"Entity: {wandb_config['entity']}\n"
                f"Project: {wandb_config['project']}\n"
                f"Name: {training_config['run_name']}\n"
            )

            os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
            os.environ["WANDB_DISABLED"] = "false"
            os.environ["WANDB_PROJECT"] = wandb_config["project"]
            os.environ["WANDB_ENTITY"] = wandb_config["entity"]

            if resume is True and wandb_config["run_id"] is not None:
                os.environ["WANDB_RESUME"] = "must"
                os.environ["WANDB_RUN_ID"] = wandb_config["run_id"]
            else:
                os.environ["WANDB_RESUME"] = "never"
                os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()


            HuggingFaceTrainer.__logger.info(f"Run id: {os.environ['WANDB_RUN_ID']}\n")
            return ["wandb", "tensorboard"]
        else:
            os.environ["WANDB_DISABLED"] = "true"
            HuggingFaceTrainer.__logger.info("Wandb is not running!")
            return ["tensorboard"]

    '''@staticmethod
    def hyperparameter_space(trial: optuna.Trial):

        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
            "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
            'num_train_epochs': trial.suggest_int(low=2, high=50, log=True)
        }
    '''

    @staticmethod
    def setup_model(
        model_config,
        model_path,
        pretrained_model_path,
        resume,
        tie_encoder_decoder,
        tokenizer,
    ):
        '''
        Function which allow to set up correcly the pre-trained checkpoints for our purposes, depending on different stages of fine-tuning,
        if starting it from scratch or resuming it from other checkpoints.
        :param model_config: configuration parameters for the model
        :param model_path: path where you want to save the pre-trained checkpoints
        :param pretrained_model_path: path where you have your saved pre-trained checkpoints
        :param resume: True or False according to the fine-tuning stage
        :param tie_encoder_decoder: if you want to let the encoder and the decoder to share the parameters
        :param tokenizer: type of tokenizer we are using
        :return: The model set completely
        '''

        if resume:

            #you resume from some fine-tuned checkpoints
            model = EncoderDecoderModel.from_pretrained(pretrained_model_path)
            HuggingFaceTrainer.__logger.info(f"Resuming from: {pretrained_model_path}.")

        #if you have the pre-trained model saved locally, you just provide the function with the directory
        elif pretrained_model_path is not None and model_path is None:

            model = EncoderDecoderModel.from_pretrained(pretrained_model_path)
            HuggingFaceTrainer.__logger.info(
                f"Model loaded from: {pretrained_model_path}."
            )
        # if you don't have the pre-trained model saved locally and you want to save it to a specific directory
        # by specifying tie_encoder_decoder variable you decide if to let the weights to be shared between Encoder and Decoder
        elif pretrained_model_path is not None:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                model_path, model_path, tie_encoder_decoder=tie_encoder_decoder
            )
            model.save_pretrained(pretrained_model_path)
            HuggingFaceTrainer.__logger.info(f"Model stored at:{pretrained_model_path}")

        else:
            raise ValueError(
                "Please provide either `pretrained_model_path` or `model_path` and `pretrained_model_path`."
            )


        #potrei cambiare la vocab_size per provare
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



    @staticmethod
    def train(
        ds_path: str,
        training_output_path: str,
        training_config: Dict,
        model_config: Dict,
        save_model_path: str,
        tokenizer_id: str,
        tie_encoder_decoder: bool,
        wandb_config: Optional[Dict] = None,
        hf_logging_enabled: bool = True,
        resume: bool = False,
        pretrained_model_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        '''
        The actual training function which allows to proceed and fine-tune the model on our specific data.
        '''
        gc.enable()

        #set up of tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, config= AutoConfig.from_pretrained(tokenizer_id))

        #set up of types of metrics we would like to compute
        compute_metrics = functools.partial(
            HuggingFaceTrainer.compute_metrics, tokenizer
        )

        #set up of wandb
        report_to = HuggingFaceTrainer.setup_wandb(
            resume, training_config, wandb_config
        )
        wandb.init(project= wandb_config['project'], entity= wandb_config['entity'], id = wandb_config['run_id'])

        #loading of the two datasets
        train_ds, eval_ds = HuggingFaceTrainer.load_dataset(ds_path)


        if hf_logging_enabled:
            HuggingFaceTrainer.__logger.info("HF logging activated.")
            hf_logging.set_verbosity_info()

        #the model here is set up, thanks to the function implemented in the other script
        model = HuggingFaceTrainer.setup_model(
            model_config,
            model_path,
            pretrained_model_path,
            resume,
            tie_encoder_decoder,
            tokenizer,
        )

        #instantiate a data collator to be given as additional argument to the model
        d_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model, padding= 'max_length')

        #set up the Seq2Seq Training arguments
        training_arguments = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="epoch",
            num_train_epochs=training_config["num_train_epochs"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            fp16=training_config["fp16"] if torch.cuda.is_available() else False,
            output_dir=training_output_path,
            overwrite_output_dir=False,
            logging_steps=training_config["logging_steps"],
            save_strategy = 'epoch',
            warmup_ratio = 0.10,
            report_to=report_to,
            run_name=training_config["run_name"],
            ignore_data_skip=resume,
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            save_total_limit=training_config["save_total_limit"],
        )


        #set uo the Seq2SeqTrainer, the one in charge of coordinating the training phase.
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_arguments,
            compute_metrics=compute_metrics,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator= d_collator
        )

        if resume:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        trainer.save_model(save_model_path)

        if wandb_config is not None:
            wandb.finish()

        torch.cuda.empty_cache()