import functools
from typing import Dict, List, Tuple, Union
import pandas as pd
from transformers import AutoTokenizer
from settings import *
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict


class HuggingFaceDataset:

    @staticmethod
    def hf_dataset(
        path1,
        path2,
        remove_columns_list: List[str],
        identifier: str,
        batch_size: int = 8,
    ) -> Tuple[Union[Dataset, DatasetDict], Union[Dataset, DatasetDict]]:
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

        dataset1 = data.train_test_split(shuffle=True, test_size=0.05)
        train_ds = dataset1["train"].shuffle(seed=42)
        test_ds = dataset1["test"]
        dataset1.save_to_disk(HF_DATASETS + '/finalized_df_1')
        train_ds.to_csv(HF_DATASETS + '/train_fin_1.csv', index=False)
        test_ds.to_csv(HF_DATASETS + '/test_fin_1.csv', index=False)
        return


    @staticmethod
    def __process(auto_tokenizer, batch: Dict):
        tokenizer = auto_tokenizer
        encoder_max_length = 40
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



'''#with the code below I create the train and test split and I save it in the local folder

df_prova = pd.read_csv(HF_DATASETS + '/finilized_dataset_1.csv')
colonna_complessa = [str(riga) for riga in list(df_prova['Normal'])]
colonna_semplice = [str(riga) for riga in list(df_prova['Simple'])]

dataframe_prova = pd.DataFrame({"Normal": colonna_complessa, "Simple": colonna_semplice})
HuggingFaceDataset.get_train_test_csv(dataframe_prova)'''