import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
USER_PATTERN = re.compile(r"/u/\w+|u/\w+")
SUBREDDIT_PATTERN = re.compile(r"/r/\w+|r/\w+")
MULTISPACE_PATTERN = re.compile(r"\s+")


def load_goemotions():
    """
    Load GoEmotions from Hugging Face and return dataset metadata.
    """
    dataset = load_dataset("go_emotions")
    label_names = dataset["train"].features["labels"].feature.names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    return dataset, label_names, id2label, label2id


def dataset_to_dataframes(dataset, id2label):
    """
    Convert splits to pandas DataFrames and add useful columns.
    """
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_df = add_basic_columns(train_df, id2label)
    val_df = add_basic_columns(val_df, id2label)
    test_df = add_basic_columns(test_df, id2label)

    return train_df, val_df, test_df


def add_basic_columns(df, id2label):
    """
    Add helper columns for EDA / preprocessing.
    """
    df = df.copy()
    df["num_labels"] = df["labels"].apply(len)
    df["char_len"] = df["text"].astype(str).apply(len)
    df["word_len"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    df["label_names"] = df["labels"].apply(lambda ids: [id2label[i] for i in ids])
    return df


def basic_clean_text(
    text,
    lowercase=True,
    remove_urls=True,
    remove_user_refs=False,
    remove_subreddit_refs=False,
):
    """
    Lightweight text cleaning.
    """
    text = str(text)

    if remove_urls:
        text = URL_PATTERN.sub(" ", text)
    if remove_user_refs:
        text = USER_PATTERN.sub(" ", text)
    if remove_subreddit_refs:
        text = SUBREDDIT_PATTERN.sub(" ", text)

    text = text.strip()
    text = MULTISPACE_PATTERN.sub(" ", text)

    if lowercase:
        text = text.lower()

    return text


def preprocess_for_tfidf(text):
    """
    More aggressive cleaning for classical ML.
    """
    return basic_clean_text(
        text,
        lowercase=True,
        remove_urls=True,
        remove_user_refs=True,
        remove_subreddit_refs=True,
    )


def preprocess_for_transformer(text):
    """
    Minimal cleaning for transformer models.
    """
    return basic_clean_text(
        text,
        lowercase=False,
        remove_urls=True,
        remove_user_refs=False,
        remove_subreddit_refs=False,
    )


def apply_preprocessing(train_df, val_df, test_df):
    """
    Add cleaned text columns.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["text_clean_tfidf"] = train_df["text"].apply(preprocess_for_tfidf)
    val_df["text_clean_tfidf"] = val_df["text"].apply(preprocess_for_tfidf)
    test_df["text_clean_tfidf"] = test_df["text"].apply(preprocess_for_tfidf)

    train_df["text_clean_transformer"] = train_df["text"].apply(preprocess_for_transformer)
    val_df["text_clean_transformer"] = val_df["text"].apply(preprocess_for_transformer)
    test_df["text_clean_transformer"] = test_df["text"].apply(preprocess_for_transformer)

    return train_df, val_df, test_df


def remove_empty_rows(df, text_col="text_clean_transformer"):
    """
    Remove rows with empty cleaned text.
    """
    df = df.copy()
    df = df[df[text_col].astype(str).str.strip() != ""].reset_index(drop=True)
    return df


def make_binary_label_matrix(df, label_names):
    """
    Convert list-of-label-ids into multilabel binary matrix.
    """
    mlb = MultiLabelBinarizer(classes=list(range(len(label_names))))
    y = mlb.fit_transform(df["labels"])
    y_df = pd.DataFrame(y, columns=label_names)
    return y_df, mlb


def transform_binary_label_matrix(df, mlb, label_names):
    """
    Transform another split using an already-fit MultiLabelBinarizer.
    """
    y = mlb.transform(df["labels"])
    y_df = pd.DataFrame(y, columns=label_names)
    return y_df


def subsample_rows(df, n_samples, random_state=42):
    """
    Simple random row subsample.
    """
    n_samples = min(n_samples, len(df))
    return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)


def subsample_multilabel_by_label_coverage(df, label_names, samples_per_label=8, random_state=42):
    """
    Build a smaller training subset while trying to keep coverage across labels.

    Strategy:
    - For each label, sample up to `samples_per_label` examples that contain it.
    - Union all selected rows.
    - Shuffle result.
    """
    rng = np.random.default_rng(random_state)

    selected_indices = set()

    for label_id in range(len(label_names)):
        label_rows = df[df["labels"].apply(lambda ids: label_id in ids)]
        if len(label_rows) == 0:
            continue

        take = min(samples_per_label, len(label_rows))
        chosen = rng.choice(label_rows.index.to_numpy(), size=take, replace=False)
        selected_indices.update(chosen.tolist())

    sampled_df = df.loc[sorted(selected_indices)].sample(frac=1, random_state=random_state)
    return sampled_df.reset_index(drop=True)


def save_dataframe(df, output_path):
    """
    Save dataframe based on file extension.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError("Supported file types are .csv and .parquet")


def save_clean_splits(train_df, val_df, test_df, output_dir="data"):
    """
    Save cleaned full splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe(train_df, output_dir / "train_clean.csv")
    save_dataframe(val_df, output_dir / "validation_clean.csv")
    save_dataframe(test_df, output_dir / "test_clean.csv")


def save_subsampled_train_sets(train_df, label_names, output_dir="data", sample_sizes=(8, 16, 32), random_state=42):
    """
    Save multiple few-shot subsampled training sets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for k in sample_sizes:
        sampled_df = subsample_multilabel_by_label_coverage(
            train_df,
            label_names=label_names,
            samples_per_label=k,
            random_state=random_state,
        )
        save_dataframe(sampled_df, output_dir / f"train_subsample_{k}.csv")
