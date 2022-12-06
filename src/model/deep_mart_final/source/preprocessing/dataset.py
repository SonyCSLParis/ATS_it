import codecs
import functools
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext import data
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig

class DatasetHelper:
    @classmethod
    def concat_and_split(
        csv_paths: List[str],
        train_csv_path: str,
        test_csv_path: str,
        shuffle: bool = False,
    ):
        dataframes: List[pd.DataFrame] = []
        for path in tqdm(csv_paths):
            df = pd.read_csv(path)
            dataframes.append(df)
        final_df = pd.concat(dataframes, ignore_index=True)
        if shuffle:
            final_df = final_df.sample(frac=1)

        train_share = int(len(final_df) * 0.8)
        train = final_df[:train_share]
        test = final_df[train_share + 1 :]

        train.to_csv(train_csv_path)
        test.to_csv(test_csv_path)

    @staticmethod
    def train_test_split(
        train_size: float,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
        shuffle: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if df is not None:
            print("Using DataFrame provided.")
        elif csv_path is not None:
            df = pd.read_csv(csv_path, index_col=0)
            print("Loaded DataFrame from CSV.")
        else:
            raise ValueError("Neither a DataFrame nor a CSV path is provided.")
        if shuffle:
            df = df.sample(frac=1)
        train_share = int(len(df) * train_size)
        train_df = df[:train_share]
        test_df = df[train_share + 1 :]
        return train_df, test_df

    @staticmethod
    def concat_csvs(csv_paths: List[str], output_csv_path: str, shuffle: bool = False):
        dataframes: List[pd.DataFrame] = []
        for path in csv_paths:
            df = pd.read_csv(path)
            dataframes.append(df)
        final_df = pd.concat(dataframes, ignore_index=True)
        if shuffle:
            final_df = final_df.sample(frac=1)
        final_df.to_csv(output_csv_path, index=False)

    @staticmethod
    def concat_split(
        middle_csv_path: str,
        to_split_csv_path: str,
        output_csv_path: str,
        first_split: float = 0.8,
    ):
        middle_df = pd.read_csv(middle_csv_path)
        to_split_df = pd.read_csv(to_split_csv_path)
        print("middle_df.shape", middle_df.shape)
        print("to_split_df.shape", to_split_df.shape)
        to_split_df = to_split_df.sample(frac=1)
        split_row = int(len(to_split_df) * first_split)
        top_split = to_split_df.iloc[:split_row]
        bottom_split = to_split_df.iloc[split_row:]

        df = pd.concat([top_split, middle_df, bottom_split])
        if "Unnamed: 0" in df.columns:
            df.drop(["Unnamed: 0"], axis=1, inplace=True)

        df.reset_index(drop=True)
        print("df.shape", df.shape)
        print(df.head())
        print(df.tail())
        df.to_csv(output_csv_path, index=False)

    @staticmethod
    def data_loaders_from_datasets(
        train_dataset: Dataset, val_dataset: Dataset, batch_size: int, pin_memory=False
    ) -> Tuple[DataLoader, DataLoader]:
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        return train_data_loader, val_data_loader


class HuggingFaceDataset:

    @staticmethod
    def hf_dataset(
        path1,
        path2,
        remove_columns_list: List[str],
        identifier: str,
        batch_size: int = 8,
    ) -> Dataset:
        """
        :param df: Pandas DataFrame which contain a `Normal` and a `Simple` column containing sentences or short paragraphs.
        :remove_columns_list: A list of columns which should be removed. Those columns will not be a part of the dataset
        :identifier: The identifier is also known as the path for the tokenizer (e.g. `bert-base-cased`).
        :batch_size: The default batch size is set to 8. This is also the default value from Hugging Face.
        """

        train_ds = load_from_disk(path1)
        test_ds = load_from_disk(path2)


        tokenizer = AutoTokenizer.from_pretrained(identifier)
        print(f"Using {identifier} tokenizer.")

        function = functools.partial(HuggingFaceDataset.__process, tokenizer)

        dataset_tr = train_ds.map(
            function=function,
            batched=True,
            batch_size=batch_size,
            remove_columns=remove_columns_list,
        )

        dataset_ts = test_ds.map(
            function=function,
            batched=True,
            batch_size=batch_size,
            remove_columns=remove_columns_list,
        )

        dataset_tr.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
                "labels",
            ],
        )

        dataset_ts.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
                "labels",
            ],
        )

        return dataset_tr, dataset_ts

    @staticmethod
    def get_train_test_csv(df: pd.DataFrame):
        from datasets import Dataset
        data = Dataset.from_pandas(df)

        dataset1 = data.train_test_split(shuffle=True, test_size=0.10)
        train_ds = dataset1["train"].shuffle(seed=42)
        test_ds = dataset1["test"]
        dataset1.save_to_disk('/Users/francesca/Desktop/Github/Final/output/output_modello/tts')
        train_ds.to_csv('/Users/francesca/Desktop/Github/Final/output/output_modello/train_tts.csv', index=False)
        test_ds.to_csv('/Users/francesca/Desktop/Github/Final/output/output_modello/test_tts.csv', index=False)
        return


    @staticmethod
    def __process(auto_tokenizer, batch: Dict):
        tokenizer = auto_tokenizer
        encoder_max_length = 30
        decoder_max_length = 30

        inputs = tokenizer(
            batch["Normal"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
        )
        outputs = tokenizer(
            batch["Simple"],
            padding="max_length",
            truncation=True,
            max_length=decoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch



    def iterators(self):
        train_iterator, test_iterator = data.BucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=self.batch_size,
            device=self.device,
            shuffle=False,
        )
        return train_iterator, test_iterator



#with the code below I create the train and test split and I save it in the local folder

df_prova = pd.read_csv('/Users/francesca/Desktop/Github/Final/output/output_modello/tts.csv')
colonna_complessa = [str(riga) for riga in list(df_prova['Normal'])]
colonna_semplice = [str(riga) for riga in list(df_prova['Simple'])]

dataframe_prova = pd.DataFrame({"Normal": colonna_complessa, "Simple": colonna_semplice})
HuggingFaceDataset.get_train_test_csv(dataframe_prova)





