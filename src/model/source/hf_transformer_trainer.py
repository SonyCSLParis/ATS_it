import argparse
from training.hf_training import HuggingFaceTrainer
from settings import *
from src.model.source.preprocessing.dataset import get_max_length

'''Note: Parts of this code are lifted as is from those written by Christopher Lemcke.

Copyright (c) 2022, Cristopher Lemcke <github: https://github.com/chrislemke/deep-martin
'''

if __name__ == "__main__":
    '''
    Within the main, we retrieve all the arguments that were provided as input in the terminal when the code was launched.
    If we do not specify any values from the CLI, the variables will assume the default values we have set. 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--save_total_limit", type=int, default=3)
    # 4 when we use the augmented dataset, 8 when we use all the other datasets
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument(
        "--run_name", type=str, default="My first Hugging Face Seq2Seq model_deep"
    )
    parser.add_argument("--tie_encoder_decoder", type=str, default="false")
    parser.add_argument("--dataset_name", type=str, default="My first Seq2Seq dataset")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume", type=str, default="false")
    parser.add_argument("--tokenizer_id", type=str)

    parser.add_argument("--seq_max_length", type=int, default=80)
    parser.add_argument("--seq_min_length", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    #length penalty only used with beam generation type
    parser.add_argument("--length_penalty", type=float, default=-0.5)
    #beam - search decoding by calling beam_search() if num_beams > 1 and do_sample = False.
    #beam - search multinomial sampling by calling beam_sample() if num_beams > 1 and do_sample = True.
    parser.add_argument("--num_beams", type=int, default=4)

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--ds_path", type=str)
    parser.add_argument("--save_model_path", type=str)
    parser.add_argument("--training_output_path", type=str)

    parser.add_argument("--wandb_id", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_api_key", type=str)

    parser.add_argument("--hf_logging_enabled", type=str, default="true")

    args, _ = parser.parse_known_args()

    #we set up the configuration dictionary of the training parameters, which are necessary for the training phase
    training_config_dict = {
        "num_train_epochs": args.num_train_epochs,
        "fp16": True,
        "save_total_limit": args.save_total_limit,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "logging_steps": args.logging_steps,
        "run_name": args.run_name,
        "dataset": args.dataset_name,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    max_seq_length = get_max_length(CSV_FILES_PATH + args.ds_path[64:] + 'train.csv') - 80


    #we set up the configuration dictionary of model parameters, required for the model setting
    model_config_dict = {
        "max_length": max_seq_length,
        "min_length": args.seq_min_length,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "length_penalty": args.length_penalty,
        "num_beams": args.num_beams,
    }

    #we set up the configuration dictionary for the wandb parameter logging system, only if some parameters
    # that should be provided in input are not None
    wandb_config_dict = None
    if (
        args.wandb_api_key is not None
        and args.wandb_entity is not None
        and args.wandb_project is not None
    ):
        wandb_config_dict = {
            "run_id": args.wandb_id,
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "api_key": args.wandb_api_key,
        }

    print("\nmodel_path: ", args.model_path)
    print("ds_path: ", args.ds_path)
    print("training_output_path: ", args.training_output_path)
    print("save_model_path: ", args.save_model_path)

    print("\nStarting training ... have a coffee ...")

    #we call the train function which is a method of the HuggingFaceTrainer class implemented in another script
    HuggingFaceTrainer.setup_logger()
    HuggingFaceTrainer.train(
        model_path=args.model_path,
        pretrained_model_path=args.pretrained_model_path,
        ds_path=args.ds_path,
        training_output_path=args.training_output_path,
        training_config=training_config_dict,
        model_config=model_config_dict,
        save_model_path=args.save_model_path,
        wandb_config=wandb_config_dict,
        tie_encoder_decoder=True if args.tie_encoder_decoder == "true" else False,
        hf_logging_enabled=True if args.hf_logging_enabled == "true" else False,
        resume=True if args.resume == "true" else False,
        tokenizer_id=args.tokenizer_id,
    )
