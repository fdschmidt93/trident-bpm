from typing import Any, cast

import torch
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import pandas as pd
from sklearn.model_selection import train_test_split
import string

import pickle

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


def load_pkl(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def remove_duplicates(pair_df):
    columns = ["revision_id", "model_id", "unique_activities"]
    if "trace" in pair_df.columns:
        columns.append("trace")
    if "eventually_follows" in pair_df.columns:
        columns.append("eventually_follows")
    if "prefix" in pair_df.columns:
        columns.append("prefix")
    pair_df = pair_df.drop_duplicates(subset=columns)
    return pair_df


def setify(x: str):
    set_: set[str] = eval(x)
    assert isinstance(set_, set), f"Conversion failed for {x}"
    return set_


EVAL_PATH = "/work/ws/ma_fabiasch-tx/trident-dialect-copa/data/"


def deduplicate_traces(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["model_id", "revision_id", "trace"])


def split_by_model(
    df, split_sizes: list[float] = [0.2, 0.1], random_state=4
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["id"] = df["model_id"].astype(str) + "_" + df["revision_id"].astype(str)
    # model_ids = df["id"].unique()
    # train_val_ids, test_ids = train_test_split(
    #     model_ids, test_size=split_sizes[-1], random_state=random_state
    # )
    # train_ids, val_ids = train_test_split(
    #     train_val_ids, test_size=split_sizes[-2], random_state=random_state
    # )
    df["num_unique_activities"] = df["unique_activities"].apply(len)
    df = df[df["num_unique_activities"] > 1]
    with open(
        "/work/ws/ma_fabiasch-tx/trident-dialect-copa/data/bpm/train_val_test.pkl", "rb"
    ) as file:
        train_ids, val_ids, test_ids = pickle.load(file)
    train_df = df[df["id"].isin(train_ids)]
    val_df = df[df["id"].isin(val_ids)]
    test_df = df[df["id"].isin(test_ids)]
    return train_df, val_df, test_df


def load_pairs(split: str = "train") -> Dataset:
    eval_pairs = load_pkl(
        "/work/ws/ma_fabiasch-tx/trident-dialect-copa/data/bpm/eval_train_data_pairs_balanced.pkl"
    )
    eval_pairs["labels"] = ~(eval_pairs.out_of_order)
    eval_pairs = remove_duplicates(eval_pairs)
    # eval_pairs.trace = eval_pairs.trace.apply(lambda x: tuple(x))
    eval_pairs.unique_activities = eval_pairs.unique_activities.apply(setify)
    columns = [
        "model_id",
        "revision_id",
        "unique_activities",
        "labels",
        "eventually_follows",
    ]
    eval_pairs = eval_pairs.loc[:, columns]
    train, val, test = split_by_model(eval_pairs)
    if split == "train":
        return Dataset.from_pandas(train)
    elif split == "val":
        return Dataset.from_pandas(val)
    else:
        return Dataset.from_pandas(test)


def convert_next_label(line: dict):
    if not line["next"] == "[END]":
        return list(line["unique_activities"]).index(line["next"]) + 1
    else:
        return 0


def load_prefixes(split: str = "train") -> Dataset:
    eval_prefix = load_pkl(
        "/work/ws/ma_fabiasch-tx/trident-dialect-copa/data/bpm/eval_train_prefix_data.pkl"
    )
    eval_prefix["prefix"] = eval_prefix["prefix"].apply(lambda x: tuple(x))
    eval_prefix = remove_duplicates(eval_prefix)
    eval_prefix["prefix"] = eval_prefix["prefix"].apply(lambda x: list(x))
    # eval_pairs.trace = eval_pairs.trace.apply(lambda x: tuple(x))
    eval_prefix.unique_activities = eval_prefix.unique_activities.apply(setify)
    # eval_prefix is a dataframe and 'unique_activities' and 'next' are columns
    # how do I make the apply work?
    mask = ~(eval_prefix.next == "[END]")
    eval_prefix["labels"] = eval_prefix.apply(convert_next_label, axis=1)
    columns = [
        "model_id",
        "revision_id",
        "trace",
        "prefix",
        "next",
        "unique_activities",
        "labels",
    ]
    eval_prefix = eval_prefix.loc[:, columns]
    eval_prefix = eval_prefix.loc[mask]
    train, val, test = split_by_model(eval_prefix)
    # x = train['unique_activities'].apply(lambda x: len(x)).value_counts()
    # y = val['unique_activities'].apply(lambda x: len(x)).value_counts()
    # z = test['unique_activities'].apply(lambda x: len(x)).value_counts()
    # z_ = pd.concat([x, y, z], axis=1)
    # z_ = pd.concat([x, y, z], axis=0)
    # z_.columns = ["train", "val", "test"]
    # z__ = z_ / z_.sum(0)
    # z__.sort_index().cumsum(0)
    if split == "train":
        return Dataset.from_pandas(train)
    elif split == "val":
        return Dataset.from_pandas(val)
    else:
        return Dataset.from_pandas(test)


def load_split(split: str = "train") -> Dataset:
    eval_train: pd.DataFrame = load_pkl(
        "/work/ws/ma_fabiasch-tx/trident-dialect-copa/data/bpm/eval_train_data_traces_balanced.pkl"
    )
    eval_train["labels"] = ~(eval_train.apply(lambda x: len(x["label"]) > 0, axis=1))
    eval_train.trace = eval_train.trace.apply(lambda x: tuple(x))
    eval_train = remove_duplicates(eval_train)
    eval_train.trace = eval_train.trace.apply(lambda x: tuple(x))
    eval_train.unique_activities = eval_train.unique_activities.apply(setify)
    columns = ["model_id", "revision_id", "unique_activities", "trace", "labels"]
    eval_train = eval_train.loc[:, columns]
    train, val, test = split_by_model(eval_train)
    if split == "train":
        return Dataset.from_pandas(train)
    elif split == "val":
        return Dataset.from_pandas(val)
    else:
        return Dataset.from_pandas(test)


def preprocess(examples: dict, tokenizer: PreTrainedTokenizerFast) -> BatchEncoding:
    # trace: list[tuple[str]]
    # label: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, trace_ in zip(
        examples["unique_activities"], examples["trace"]
    ):
        input_ = f"Set of process activities: {str(unique_activities_)}\nProcess: {str(list(trace_))}\n: Valid: "
        inputs.append(input_)
    batch = tokenizer(inputs)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch


def preprocess_pairs(
    examples: dict, tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    # eventually_follows: list[tuple[str, str]]
    # labels: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, activities_ in zip(
        examples["unique_activities"], examples["eventually_follows"]
    ):
        input_ = f"Set of activities: {str(unique_activities_)}\nOrdered activities: {str(list(activities_))}\n: Valid: "
        inputs.append(input_)
    batch = tokenizer(inputs)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch


def preprocess_seq_clf(
    examples: dict, tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    # trace: list[tuple[str]]
    # label: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, trace_ in zip(
        examples["unique_activities"], examples["trace"]
    ):
        input_ = f"Set of process activities: {str(unique_activities_)}\nProcess: {str(list(trace_))}"
        inputs.append(input_)
    batch = tokenizer(inputs, max_length=512, truncation=True)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch


def preprocess_pairs_seq_clf(
    examples: dict, tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    # eventually_follows: list[tuple[str, str]]
    # labels: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, activities_ in zip(
        examples["unique_activities"], examples["eventually_follows"]
    ):
        input_ = f"Set of activities: {str(unique_activities_)}\nOrdered activities: {str(list(activities_))}"
        inputs.append(input_)
    batch = tokenizer(inputs, truncation=True)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch


def preprocess_next_activity(
    examples: dict, tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    # unique_activities: list[set[str]]
    # prefix: list[tuple[str]]
    # labels: list[string]

    # List of activities:
    # A. Activity
    # B. Activity
    # C. Activity
    # Which one of the above activities should follow the below sequence of activities?
    # Sequence of activites: []
    # Answer: A
    # trace: list[tuple[str]]
    # label: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, prefix_ in zip(
        examples["unique_activities"], examples["prefix"]
    ):
        string_ = "List of activites:\n0. [END]\n"
        for i, activity in enumerate(unique_activities_):
            string_ += f"{string.ascii_lowercase[i].upper()}. {activity.capitalize()}\n"

        string_ += "Which one of the above activities should follow the below sequence of activities?\n"
        string_ += f"Sequence of activities: {[p.capitalize() for p in prefix_]}\n"
        string_ += "Answer: "
        inputs.append(string_)

    batch = tokenizer(inputs)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch


def preprocess_next_activity_clf(
    examples: dict, tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    # unique_activities: list[set[str]]
    # prefix: list[tuple[str]]
    # labels: list[string]

    # List of activities:
    # A. Activity
    # B. Activity
    # C. Activity
    # Which one of the above activities should follow the below sequence of activities?
    # Sequence of activites: []
    # Answer: A
    # trace: list[tuple[str]]
    # label: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, prefix_ in zip(
        examples["unique_activities"], examples["prefix"]
    ):
        string_ = "\n0. [END]\n"
        for i, activity in enumerate(unique_activities_, 1):
            string_ += f"{i}. {activity.capitalize()}\n"
        # string_ += "Which one of the above activities should follow the below sequence of activities?\n"
        string_ += f"Sequence of activities: {[p.capitalize() for p in prefix_]}\n"
        # string_ += "Answer: "
        inputs.append(string_)
    batch = tokenizer(inputs, truncation=True)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch


# only keep unique pairs per model

# def get_trace_data(samples_per_class):
#     trace_df = pd.read_pickle(EVAL_PATH / "eval_train_data_traces.pkl")
#     trace_df["anomalous"] = trace_df.progress_apply(lambda x: len(x["label"]) > 0, axis=1)
#     # Set test_size to 0.5 to get 50% of the rows in the val set
#     trace_train_df, trace_val_df = split_by_model(trace_df, test_size=0.5, random_state=4)
#     # Sample data to have equal number of positive and negative samples
#     trace_train_df_positive = trace_train_df[~trace_train_df["anomalous"]].sample(n=samples_per_class, random_state=4)
#     trace_train_df_negative = trace_train_df[trace_train_df["anomalous"]].sample(n=samples_per_class, random_state=4)
#     trace_train_df = pd.concat([trace_train_df_positive, trace_train_df_negative])
#     # shuffle the data
#     trace_train_df = trace_train_df.sample(frac=1).reset_index(drop=True)
#     # Sample data to have equal number of positive and negative samples
#     trace_val_df_positive = trace_val_df[~trace_val_df["anomalous"]].sample(n=samples_per_class, random_state=4)
#     trace_val_df_negative = trace_val_df[trace_val_df["anomalous"]].sample(n=samples_per_class, random_state=4)
#     trace_val_df = pd.concat([trace_val_df_positive, trace_val_df_negative])
#     # shuffle the data
#     trace_val_df = trace_val_df.sample(frac=1).reset_index(drop=True)
#     return trace_train_df, trace_val_df

# def get_pair_data(samples_per_class, with_trace=False):
#     pair_df = pd.read_pickle(EVAL_PATH + "eval_train_data_pairs.pkl")
#     if not with_trace:
#         pair_df = remove_duplicates(pair_df)
#     # Set test_size to 0.5 to get 50% of the rows in the val set
#     pair_train_df, pair_val_df = split_by_model(pair_df, test_size=0.5, random_state=4)
#     # Sample data to have equal number of positive and negative samples
#     pair_train_df_positive = pair_train_df[~pair_train_df["out_of_order"]].sample(n=samples_per_class, random_state=4)
#     pair_train_df_negative = pair_train_df[pair_train_df["out_of_order"]].sample(n=samples_per_class, random_state=4)
#     pair_train_df = pd.concat([pair_train_df_positive, pair_train_df_negative])
#     # shuffle the data
#     pair_train_df = pair_train_df.sample(frac=1, random_state=4).reset_index(drop=True)
#     # Sample data to have equal number of positive and negative samples validation
#     pair_val_df_positive = pair_val_df[~pair_val_df["out_of_order"]].sample(n=samples_per_class, random_state=4)
#     pair_val_df_negative = pair_val_df[pair_val_df["out_of_order"]].sample(n=samples_per_class, random_state=4)
#     pair_val_df = pd.concat([pair_val_df_negative, pair_val_df_positive])
#     # shuffle the data
#     pair_val_df = pair_val_df.sample(frac=1, random_state=4).reset_index(drop=True)
#     return pair_train_df, pair_val_df


def preprocess_fn(
    examples: dict,
    column_names: dict,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    premise = column_names["premise"]
    question = column_names["question"]
    choice1 = column_names["choice1"]
    choice2 = column_names["choice2"]
    # Add 0 to ensure logits are properly shifted by 1 and final token predicts label (cf. DataCollator)
    text = [
        f'Premise: "{p}"\nQuestion: "{q}"\nChoice 1: "{c1}"\nChoice 2: "{c2}"\nAnswer: 0'
        for p, q, c1, c2 in zip(
            examples[premise], examples[question], examples[choice1], examples[choice2]
        )
    ]
    # Tokenize
    return tokenizer(text)


def preprocess_cls_fn(
    examples: dict,
    column_names: dict,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    premise = column_names["premise"]
    question = column_names["question"]
    choice1 = column_names["choice1"]
    choice2 = column_names["choice2"]
    # Add 0 to ensure logits are properly shifted by 1 and final token predicts label (cf. DataCollator)
    text = [
        f'Premise: "{p}"\nQuestion: "{q}"\nChoice 1: "{c1}"\nChoice 2: "{c2}"\nAnswer: '  # <- last token
        for p, q, c1, c2 in zip(
            examples[premise], examples[question], examples[choice1], examples[choice2]
        )
    ]
    # Tokenize
    return tokenizer(text)


class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, *args, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(
        self, inputs: list[dict[str, int | list[int]]], *args: Any, **kwds: Any
    ) -> BatchEncoding:
        inputs_ = {
            "input_ids": [cast(list[int], i["input_ids"]) for i in inputs],
            "attention_mask": [cast(list[int], i["attention_mask"]) for i in inputs],
        }
        flattened_labels = torch.LongTensor(
            [
                self.tokenizer.convert_tokens_to_ids(str(cast(int, i["label"]) + 1))
                for i in inputs
            ]
        )
        batch = self.tokenizer.pad(
            cast(dict[str, list[list[int]]], inputs_), return_tensors="pt", padding=True
        )
        labels = torch.full_like(
            cast(torch.Tensor, batch["input_ids"]), fill_value=-100, dtype=torch.int64
        )
        labels[:, -1] = flattened_labels
        batch["labels"] = labels
        return batch


class DataCollatorForSequenceClassification:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, *args, **kwargs) -> None:
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(
        self, inputs: list[dict[str, int | list[int]]], *args: Any, **kwds: Any
    ) -> BatchEncoding:
        inputs_ = {
            "input_ids": [cast(list[int], i["input_ids"]) for i in inputs],
            "attention_mask": [cast(list[int], i["attention_mask"]) for i in inputs],
        }
        batch = self.tokenizer.pad(
            cast(dict[str, list[list[int]]], inputs_),
            return_tensors="pt",
            max_length=512,
            padding=True,
        )
        if "labels" in inputs[0]:
            flattened_labels = torch.LongTensor([i["labels"] for i in inputs])
            batch["labels"] = flattened_labels
        return batch
