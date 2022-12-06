
from src.model.deep_mart_final.source.training.hf_training import *
#from settings import *
import optuna
import json


# customized print function
def print_custom(text):
    print('\n')
    print(text)
    print('-' * 100)


# our variable of interest
LR_MIN = 4e-5
LR_CEIL = 0.01
WD_MIN = 4e-5
WD_CEIL = 0.01
MIN_EPOCHS = 2
MAX_EPOCHS = 20
NUM_TRIALS = 10
NAME_OF_MODEL = '/BEST_HYP'



def __setup_model(
        model_config,
        model_path,
        pretrained_model_path,
        resume,
        tie_encoder_decoder,
        tokenizer,
):
    from transformers import EncoderDecoderModel

    if resume:
        model = EncoderDecoderModel.from_pretrained(pretrained_model_path)


    elif pretrained_model_path is not None and model_path is None:
        model = EncoderDecoderModel.from_pretrained(pretrained_model_path)


    elif model_path is not None:
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

    # qui nella parte destra dell'uguale c'era tokenizer.config, ora però ho messo model_deep
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





def __compute_metrics(auto_tokenizer, prediction: EvalPrediction):
    __rouge = load_metric("rouge")
    __bert_score = load_metric("bertscore")
    __meteor = load_metric("meteor")

    tokenizer = auto_tokenizer

    labels_ids = prediction.label_ids
    pred_ids = prediction.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = __rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid
    bert_score_output = __bert_score.compute(
        predictions=pred_str, references=label_str, lang="en"
    )

    meteor_output = __meteor.compute(
        predictions=pred_str, references=label_str
    )
    return {
        "bert_score_f1": round(bert_score_output["f1"][0], 4),
        "meteor_score": round(meteor_output["meteor"], 4),
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_f_measure": round(rouge_output.fmeasure, 4),
    }


my_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
compute_metrics = functools.partial(
    __compute_metrics, my_tokenizer
)

ds_path = '/Users/francesca/Desktop/Github/Final/output/output_modello/complete_df'
train_ds, eval_ds = HuggingFaceTrainer.load_dataset(ds_path)


training_config_dict = {
    "num_train_epochs": 5,
    "fp16": True,
    "eval_steps": 500,
    "warmup_steps": 5,
    "save_total_limit": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "logging_steps": 100,
    "save_steps": 100,
    "run_name": 'HuggingF Model',
    "dataset": 'Complete_dataset',
    "gradient_accumulation_steps": 1
}

model_config_dict = {
    "max_length": 30,
    "min_length": 2,
    "no_repeat_ngram_size": 3,
    "length_penalty": 0.5,
    "num_beams": 1.0,
}

print_custom('Setting up Optuna study')



def objective(trial: optuna.Trial):
    modello = __setup_model(model_config=model_config_dict,
                            model_path=None,
                            pretrained_model_path='/Users/francesca/Desktop/Github/Final/src/model/deep_mart_final/source/bert2bert',
                            resume=False,
                            tie_encoder_decoder=False,
                            tokenizer=my_tokenizer)


    training_arguments = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64, 128]),
        fp16=training_config_dict["fp16"] if torch.cuda.is_available() else False,
        output_dir='/Users/francesca/Desktop/model_deep' + '/HYPERPARAM_SEARCH',
        overwrite_output_dir=False,
        logging_steps=training_config_dict["logging_steps"],
        save_steps=training_config_dict["save_steps"],
        eval_steps=training_config_dict["eval_steps"],
        warmup_steps=training_config_dict["warmup_steps"],
        run_name=training_config_dict["run_name"],
        ignore_data_skip=True,
        gradient_accumulation_steps=training_config_dict["gradient_accumulation_steps"],
        save_total_limit=training_config_dict["save_total_limit"],
        learning_rate=trial.suggest_loguniform('learning_rate', low=LR_MIN, high=LR_CEIL),
        weight_decay=trial.suggest_loguniform('weight_decay', WD_MIN, WD_CEIL),
        num_train_epochs=trial.suggest_int('num_train_epochs', low=MIN_EPOCHS, high=MAX_EPOCHS),
    )

    trainer = Seq2SeqTrainer(
        model=modello,
        tokenizer=my_tokenizer,
        args=training_arguments,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    result = trainer.train()
    return result.training_loss


print_custom('Triggering Optuna study')
study = optuna.create_study(study_name='hf-search-paccsit', direction='minimize')
study.optimize(func=objective, n_trials=NUM_TRIALS)

print(f'The best loss is:{study.best_value}')
print()
print()
print(study.best_trial)

# This can be used to train the final model. Passed through using kwargs into the model
print_custom('Finding study best parameters')
best_lr = float(study.best_params['learning_rate'])
best_weight_decay = float(study.best_params['weight_decay'])
best_epoch = int(study.best_params['num_train_epochs'])
best_batch_size = int(study.best_params['per_device_train_batch_size'])

print_custom('Extract best study params')
print(f'The best learning rate is: {best_lr}')
print(f'The best weight decay is: {best_weight_decay}')
print(f'The best epoch is : {best_epoch}')
print(f'The best batch size is : {best_batch_size}')

print_custom('Create dictionary of the best hyperparameters')
best_hp_dict = {
    'best_learning_rate': best_lr,
    'best_weight_decay': best_weight_decay,
    'best_epoch': best_epoch,
    'best_batch_size': best_batch_size
}

with open("best_run.json", "w+") as f:
    f.write(json.dumps(study.best_params))

# ----------------------------------------------------------------------------------------------------
#                   TRAIN BASED ON OPTUNAS SELECTED HP
# ----------------------------------------------------------------------------------------------------

'''print_custom('Training the model on the custom parameters')
modello = __setup_model(model_config=model_config_dict,
                        model_path=None,
                        pretrained_model_path=BERT2BERT_DIR,
                        resume=False,
                        tie_encoder_decoder=False,
                        tokenizer=my_tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir=TRAINED_MODEL + 'BEST_HYP',
    learning_rate=best_lr,
    weight_decay=best_weight_decay,
    num_train_epochs=best_epoch,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    disable_tqdm=True)

trainer = Seq2SeqTrainer(
    model=modello,
    tokenizer=my_tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

result = trainer.train()
trainer.evaluate()

print_custom('Saving the best Optuna tuned model')

model_path = TRAINED_MODEL + NAME_OF_MODEL
modello.save_pretrained(model_path)
my_tokenizer.save_pretrained(model_path)'''
