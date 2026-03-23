import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
USER_PATTERN = re.compile(r"/u/\w+|u/\w+")
SUBREDDIT_PATTERN = re.compile(r"/r/\w+|r/\w+")
MULTISPACE_PATTERN = re.compile(r"\s+")
ALNUM_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9]+$")


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


def _import_nltk_modules():
    """Import NLTK and return required symbols with a clear error if unavailable."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
    except Exception as exc:
        raise ImportError(
            "NLTK preprocessing requested, but NLTK is not available. "
            "Install `nltk` in your notebook setup cell first."
        ) from exc

    return nltk, stopwords, WordNetLemmatizer, word_tokenize


def ensure_nltk_resources(*, download: bool = False, use_lemmatizer: bool = False):
    """
    Ensure required NLTK resources are available.

    If `download=True`, missing resources are downloaded.
    Otherwise, raises with an actionable error message.
    """
    nltk, _, _, _ = _import_nltk_modules()

    required = [
        ("tokenizers/punkt", "punkt"),
        # Some NLTK versions require punkt_tab for word_tokenize.
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    if use_lemmatizer:
        required.extend(
            [
                ("corpora/wordnet", "wordnet"),
                ("corpora/omw-1.4", "omw-1.4"),
            ]
        )

    missing = []
    for resource_path, package_name in required:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(package_name)

    if not missing:
        return

    if download:
        for package_name in missing:
            nltk.download(package_name, quiet=True)
        return

    raise RuntimeError(
        "Missing NLTK resources: "
        + ", ".join(sorted(set(missing)))
        + ". Run ensure_nltk_resources(download=True) first."
    )


def preprocess_with_nltk(
    text,
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_user_refs: bool = True,
    remove_subreddit_refs: bool = True,
    remove_stopwords: bool = True,
    remove_non_alnum_tokens: bool = True,
    lemmatize: bool = False,
    download_nltk_resources: bool = False,
):
    """
    NLTK-based preprocessing: tokenization + optional stopword removal/lemmatization.
    """
    cleaned = basic_clean_text(
        text,
        lowercase=lowercase,
        remove_urls=remove_urls,
        remove_user_refs=remove_user_refs,
        remove_subreddit_refs=remove_subreddit_refs,
    )

    ensure_nltk_resources(download=download_nltk_resources, use_lemmatizer=lemmatize)
    _, stopwords, WordNetLemmatizer, word_tokenize = _import_nltk_modules()

    tokens = word_tokenize(cleaned)

    if remove_non_alnum_tokens:
        tokens = [tok for tok in tokens if ALNUM_TOKEN_PATTERN.match(tok)]

    if remove_stopwords:
        stopword_set = set(stopwords.words("english"))
        tokens = [tok for tok in tokens if tok.lower() not in stopword_set]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    return " ".join(tokens)


def preprocess_for_tfidf(text):
    """
    More aggressive regex cleaning for classical ML.
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
    Minimal regex cleaning for transformer models.
    """
    return basic_clean_text(
        text,
        lowercase=False,
        remove_urls=True,
        remove_user_refs=False,
        remove_subreddit_refs=False,
    )


def apply_preprocessing(
    train_df,
    val_df,
    test_df,
    *,
    mode: str = "regex",
    use_lemmatizer: bool = False,
    download_nltk_resources: bool = False,
):
    """
    Add cleaned text columns.

    Modes:
    - "regex" (default): use existing regex cleaners
    - "nltk": use NLTK tokenization + stopword pipeline
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    mode = mode.strip().lower()

    if mode == "regex":
        train_df["text_clean_tfidf"] = train_df["text"].apply(preprocess_for_tfidf)
        val_df["text_clean_tfidf"] = val_df["text"].apply(preprocess_for_tfidf)
        test_df["text_clean_tfidf"] = test_df["text"].apply(preprocess_for_tfidf)

        train_df["text_clean_transformer"] = train_df["text"].apply(preprocess_for_transformer)
        val_df["text_clean_transformer"] = val_df["text"].apply(preprocess_for_transformer)
        test_df["text_clean_transformer"] = test_df["text"].apply(preprocess_for_transformer)

        return train_df, val_df, test_df

    if mode == "nltk":
        tfidf_kwargs = {
            "lowercase": True,
            "remove_urls": True,
            "remove_user_refs": True,
            "remove_subreddit_refs": True,
            "remove_stopwords": True,
            "remove_non_alnum_tokens": True,
            "lemmatize": use_lemmatizer,
            "download_nltk_resources": download_nltk_resources,
        }
        transformer_kwargs = {
            "lowercase": False,
            "remove_urls": True,
            "remove_user_refs": False,
            "remove_subreddit_refs": False,
            "remove_stopwords": False,
            "remove_non_alnum_tokens": False,
            "lemmatize": False,
            "download_nltk_resources": download_nltk_resources,
        }

        train_df["text_clean_tfidf"] = train_df["text"].apply(
            lambda x: preprocess_with_nltk(x, **tfidf_kwargs)
        )
        val_df["text_clean_tfidf"] = val_df["text"].apply(
            lambda x: preprocess_with_nltk(x, **tfidf_kwargs)
        )
        test_df["text_clean_tfidf"] = test_df["text"].apply(
            lambda x: preprocess_with_nltk(x, **tfidf_kwargs)
        )

        train_df["text_clean_transformer"] = train_df["text"].apply(
            lambda x: preprocess_with_nltk(x, **transformer_kwargs)
        )
        val_df["text_clean_transformer"] = val_df["text"].apply(
            lambda x: preprocess_with_nltk(x, **transformer_kwargs)
        )
        test_df["text_clean_transformer"] = test_df["text"].apply(
            lambda x: preprocess_with_nltk(x, **transformer_kwargs)
        )

        return train_df, val_df, test_df

    raise ValueError("Unsupported mode. Use 'regex' or 'nltk'.")


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


def _make_labelset_stratify_keys(
    labels_series: pd.Series,
    *,
    min_count_for_stratify: int = 2,
    rare_bucket: str = "__RARE__",
) -> pd.Series:
    """
    Build stratification keys from multilabel sets.

    Uses sorted label-tuples converted to strings and buckets rare combinations.
    """
    key_series = labels_series.apply(lambda ids: "|".join(str(i) for i in sorted(ids)))
    counts = key_series.value_counts()
    valid_keys = set(counts[counts >= min_count_for_stratify].index)
    return key_series.apply(lambda key: key if key in valid_keys else rare_bucket)


def subsample_train_fraction(
    train_df: pd.DataFrame,
    fraction: float,
    *,
    random_state: int = 42,
    strategy: str = "labelset_stratified",
    label_col: str = "labels",
) -> pd.DataFrame:
    """
    Subsample a training dataframe by fraction.

    Strategies:
    - "labelset_stratified": approximate multilabel stratification using labelset keys
    - "random": simple random sampling
    """
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1].")

    n_total = len(train_df)
    n_samples = max(1, int(round(n_total * fraction)))

    if n_samples >= n_total:
        return train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    strategy = strategy.strip().lower()

    if strategy == "random":
        return train_df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

    if strategy != "labelset_stratified":
        raise ValueError("Unsupported strategy. Use 'labelset_stratified' or 'random'.")

    try:
        stratify_keys = _make_labelset_stratify_keys(train_df[label_col])
        if stratify_keys.nunique() < 2:
            raise ValueError("Not enough distinct stratification buckets.")

        sampled_df, _ = train_test_split(
            train_df,
            train_size=n_samples,
            random_state=random_state,
            stratify=stratify_keys,
            shuffle=True,
        )
        return sampled_df.reset_index(drop=True)
    except ValueError:
        # Fallback for very small/rare label distributions.
        return train_df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)


def generate_fraction_subsamples(
    train_df: pd.DataFrame,
    *,
    fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.25, 0.50),
    seeds: Iterable[int] = (42,),
    strategy: str = "labelset_stratified",
    label_col: str = "labels",
) -> Dict[tuple, pd.DataFrame]:
    """
    Generate subsamples for each (fraction, seed) pair.

    Returns dict keyed by (fraction, seed).
    """
    subsamples: Dict[tuple, pd.DataFrame] = {}

    for seed in seeds:
        for fraction in fractions:
            subsamples[(float(fraction), int(seed))] = subsample_train_fraction(
                train_df,
                float(fraction),
                random_state=int(seed),
                strategy=strategy,
                label_col=label_col,
            )

    return subsamples


def _fraction_to_pct_tag(fraction: float) -> str:
    pct = int(round(fraction * 100))
    return f"{pct}pct"


def normalize_multilabel_column(df: pd.DataFrame, label_col: str = "labels") -> pd.DataFrame:
    """
    Normalize multilabel column values to plain Python int lists.

    This prevents CSV serialization like "[ 3 10]" (numpy array style) and
    ensures downstream `ast.literal_eval` works consistently.
    """
    df = df.copy()

    if label_col not in df.columns:
        return df

    def _normalize_value(value):
        if isinstance(value, list):
            return [int(x) for x in value]
        if isinstance(value, tuple):
            return [int(x) for x in value]
        if isinstance(value, np.ndarray):
            return [int(x) for x in value.tolist()]

        if isinstance(value, str):
            s = value.strip()
            if s.startswith("[") and s.endswith("]"):
                body = s[1:-1].strip()
                if body == "":
                    return []
                if "," in body:
                    tokens = [tok.strip() for tok in body.split(",") if tok.strip()]
                else:
                    tokens = [tok for tok in body.split() if tok]
                try:
                    return [int(tok) for tok in tokens]
                except ValueError:
                    return value

        return value

    df[label_col] = df[label_col].apply(_normalize_value)
    return df


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

    train_df = normalize_multilabel_column(train_df, label_col="labels")
    val_df = normalize_multilabel_column(val_df, label_col="labels")
    test_df = normalize_multilabel_column(test_df, label_col="labels")

    save_dataframe(train_df, output_dir / "train_clean.csv")
    save_dataframe(val_df, output_dir / "validation_clean.csv")
    save_dataframe(test_df, output_dir / "test_clean.csv")


def save_standard_splits(train_df, val_df, test_df, output_dir="data"):
    """
    Save standard processed splits expected by downstream notebooks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = normalize_multilabel_column(train_df, label_col="labels")
    val_df = normalize_multilabel_column(val_df, label_col="labels")
    test_df = normalize_multilabel_column(test_df, label_col="labels")

    save_dataframe(train_df, output_dir / "train.csv")
    save_dataframe(val_df, output_dir / "val.csv")
    save_dataframe(test_df, output_dir / "test.csv")


def save_subsampled_train_sets(train_df, label_names, output_dir="data", sample_sizes=(8, 16, 32), random_state=42):
    """
    Save multiple few-shot subsampled training sets (legacy helper).
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


def save_fraction_subsamples(
    train_df: pd.DataFrame,
    *,
    output_dir: str = "data",
    fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.25, 0.50),
    seeds: Iterable[int] = (42,),
    strategy: str = "labelset_stratified",
    label_col: str = "labels",
    canonical_seed: int = 42,
    write_seeded_files: bool = True,
    write_canonical_aliases: bool = True,
):
    """
    Save fraction-based training subsets for one or more seeds.

    File naming:
    - Seeded: train_10pct_seed42.csv
    - Canonical alias (for canonical seed): train_10pct.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = normalize_multilabel_column(train_df, label_col=label_col)

    generated = generate_fraction_subsamples(
        train_df,
        fractions=fractions,
        seeds=seeds,
        strategy=strategy,
        label_col=label_col,
    )

    for (fraction, seed), sampled_df in generated.items():
        pct_tag = _fraction_to_pct_tag(fraction)

        if write_seeded_files:
            seeded_path = output_dir / f"train_{pct_tag}_seed{seed}.csv"
            save_dataframe(sampled_df, seeded_path)

        if write_canonical_aliases and int(seed) == int(canonical_seed):
            canonical_path = output_dir / f"train_{pct_tag}.csv"
            save_dataframe(sampled_df, canonical_path)

    return generated
