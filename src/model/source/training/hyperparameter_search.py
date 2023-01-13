import json
from settings import *
from hf_training import *
'''
In this script, the necessary pipeline was set up to conduct an optimisation of the hyper-parameter space.
Then, thanks to this optimisation process, we should find the best parameters with which to finally draw our best model.
'''
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
MAX_EPOCHS = 50
NUM_TRIALS = 50
NAME_OF_MODEL = '/BEST_HYP'


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


#ds_path = HF_DATASETS + '/finalized_df_1'
ds_path = HF_DATASETS + '/augmented'
train_ds, eval_ds = HuggingFaceTrainer.load_dataset(ds_path)


training_config_dict = {
    "fp16": True,
    "save_total_limit": 3,
    "run_name": 'Hyperparameter_Search',
    "dataset": 'Augmented_dataset',
    "gradient_accumulation_steps": 1
}

model_config_dict = {
    "max_length": 70,
    "min_length": 3,
    "no_repeat_ngram_size": 3,
    "length_penalty": -0.5,
    "num_beams": 4,
}


wandb_config_dict = { "run_id": 'trial_nuovo',
                      "entity": 'sony',
                      "project":'Automatic_Text_Simplification',
                      "api_key": 'REDACTED',
}

my_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
compute_metrics = functools.partial(
            HuggingFaceTrainer.compute_metrics, my_tokenizer
        )

report_to = HuggingFaceTrainer.setup_wandb(
            False, training_config_dict, wandb_config_dict
        )


print_custom('Setting up Optuna study')


wandb.init(project= wandb_config_dict['project'], entity= wandb_config_dict['entity'], id = wandb_config_dict['run_id'], resume='allow')

def objective(trial: optuna.Trial):
    modello = HuggingFaceTrainer.setup_model(model_config=model_config_dict,
                            model_path=None,
                            pretrained_model_path= BERT2BERT_CASED_DIR,
                            resume=False,
                            tie_encoder_decoder=False,
                            tokenizer=my_tokenizer)


    training_arguments = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64, 128]),
        per_device_eval_batch_size=trial.suggest_categorical("per_device_eval_batch_size", [4, 8, 16, 32, 64, 128]),
        fp16=training_config_dict["fp16"] if torch.cuda.is_available() else False,
        max_grad_norm=0.5,
        output_dir= TRAINED_MODEL + '/HYPERPARAM_SEARCH',
        overwrite_output_dir=False,
        #logging_steps=training_config_dict["logging_steps"],
        run_name=training_config_dict["run_name"],
        ignore_data_skip=True,
        save_strategy='epoch',
        report_to = report_to,
        warmup_ratio=0.10,
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
#                   TRAIN BASED ON OPTUNA SELECTED HP
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
