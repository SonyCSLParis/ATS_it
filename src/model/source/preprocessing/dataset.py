import functools
from typing import Dict, List, Tuple, Union

import datasets
import pandas as pd
from transformers import AutoTokenizer
from settings import *
from datasets import  load_from_disk, Dataset, DatasetDict

'''Note: Parts of this code are lifted as is from those written by Christopher Lemke.
Copyright (c) 2022, Cristopher Lemke <github: https://github.com/chrislemke/deep-martin
'''


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
        :param path1 and path2: path for the Dataset HF format for train and test dataset
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
    def get_train_test_csv(df: pd.DataFrame, output_name, train_name,val_name, test_name):
        '''
        This function has the objactive of saving locally the train and test split of our input dataset.
        :param df: the input dataset, in format .csv
        :return: the Hugging Face Dataset type, it saves it in the correct local directory
        '''
        from datasets import Dataset
        data = Dataset.from_pandas(df)

        dataset1 = data.train_test_split(shuffle=True, test_size=0.1)
        train_ds = dataset1["train"].shuffle(seed=42)
        test_valid = dataset1["test"].train_test_split(test_size = 0.5, shuffle = True)
        val_ds = test_valid['train']
        test_ds = test_valid['test']

        dataset_finale = datasets.DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds})


        dataset_finale.save_to_disk(HF_DATASETS + output_name)
        train_ds.to_csv(CSV_FILES_PATH + train_name, index=False)
        val_ds.to_csv(CSV_FILES_PATH + val_name, index = False)
        test_ds.to_csv(CSV_FILES_PATH + test_name, index=False)
        return


    @staticmethod
    def __process(auto_tokenizer, batch: Dict):
        '''
        This function allows to process the dataset and prepare the batch of sentences to be processed by the HF model (e.g., a transformer)
        :param auto_tokenizer: type of tokenizer used
        :param batch: our original sentences in batch
        :return: processed sentences in batch
        '''
        tokenizer = auto_tokenizer
        encoder_max_length = 80
        decoder_max_length = 80

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

df1 = pd.read_csv(CSV_FILES_PATH + 'finalized_dataset/finalized_df.csv')
df2 = pd.read_csv(CSV_FILES_PATH + '/augmented/augmented_dataset.csv')
df3 = pd.read_csv(CSV_FILES_PATH + '/paccssit/paccss_only_1.csv')

for df in [(df1, '/finalized_dataset'),(df2, '/augmented'),(df3, '/paccssit')]:

    colonna_complessa = [str(riga) for riga in list(df[0]['Normal'])]
    colonna_semplice = [str(riga) for riga in list(df[0]['Simple'])]

    dataframe_prova = pd.DataFrame({"Normal": colonna_complessa, "Simple": colonna_semplice})
    HuggingFaceDataset.get_train_test_csv(dataframe_prova, df[1], df[1] + '/train.csv', df[1] + '/val.csv', df[1] + '/test.csv')'''