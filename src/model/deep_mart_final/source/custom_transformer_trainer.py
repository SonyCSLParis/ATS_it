import argparse

from transformers import AutoTokenizer
from src.PACCS.settings import *
from src.model.deep_mart_final.source.training.custom_transformer_training import TransformerTrainer
from src.model.deep_mart_final.source.analysis.evaluator import CTransformerEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=int, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--print_every", type=int, default=300)
    parser.add_argument("--lr", type=int, default=0.0001)
    parser.add_argument("--load_weights")
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--checkpoint", type=str, default="false")
    parser.add_argument("--save_model_path", type=str)
    parser.add_argument("--HF_tokenizer", type=str, default="dbmdz/bert-base-italian-xxl-cased")
    parser.add_argument("--loss_plot_save_path", type=str)

    args, _ = parser.parse_known_args()

    tokenizer = AutoTokenizer.from_pretrained(args.HF_tokenizer)
    TransformerTrainer.start_training(
        args,
        dataset_path=args.ds_path,
        vocab_size=tokenizer.vocab_size,
        train_file=args.train_file,
        test_file=args.test_file,
        save_model_path=args.save_model_path,
    )


'''eval_costum = CTransformerEvaluator(arguments=args,
                                    vocab_size=tokenizer.vocab_size,
                                    eval_dataset_path = OUTPUT_DIR + 'df_test_ultimated.csv',
                                    model_path ='/model_deep/trained_model/custom_3_epochs/model_deep.pt',
                                    tokenizer_path = "dbmdz/bert-base-italian-xxl-cased")

eval_costum.evaluate()'''