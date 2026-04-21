# GoEmotions labels grouped into Ekman's six basic emotions
# labels that do not cleanly map to Ekman are assigned to "other"

EKMAN_6_GROUP_MAPPING = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "joy",
    "confusion": "other",
    "curiosity": "other",
    "desire": "joy",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "other",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "other",
}


# coarser sentiment-style grouping used in many class-balance analyses

POS_NEG_AMBIGUOUS_NEUTRAL_MAPPING = {
    "admiration": "positive",
    "amusement": "positive",
    "anger": "negative",
    "annoyance": "negative",
    "approval": "positive",
    "caring": "positive",
    "confusion": "ambiguous",
    "curiosity": "ambiguous",
    "desire": "positive",
    "disappointment": "negative",
    "disapproval": "negative",
    "disgust": "negative",
    "embarrassment": "negative",
    "excitement": "positive",
    "fear": "negative",
    "gratitude": "positive",
    "grief": "negative",
    "joy": "positive",
    "love": "positive",
    "nervousness": "negative",
    "optimism": "positive",
    "pride": "positive",
    "realization": "ambiguous",
    "relief": "positive",
    "remorse": "negative",
    "sadness": "negative",
    "surprise": "ambiguous",
    "neutral": "neutral",
}


SUPPORTED_SCHEMES = {
    "ekman",
    "ekman6",
    "ekman_6",
    "pos_neg_ambiguous_neutral",
    "sentiment4",
}


def get_label_group_mapping(scheme):
    # allow a few aliases so notebooks can pass human-friendly names.
    normalized = scheme.strip().lower()

    if normalized in {"ekman", "ekman6", "ekman_6"}:
        return EKMAN_6_GROUP_MAPPING.copy()
    if normalized in {"pos_neg_ambiguous_neutral", "sentiment4"}:
        return POS_NEG_AMBIGUOUS_NEUTRAL_MAPPING.copy()

    raise ValueError(
        f"Unsupported scheme '{scheme}'. Supported schemes: {sorted(SUPPORTED_SCHEMES)}"
    )


def validate_label_group_mapping(label_names, mapping, *, allow_unmapped=False):
    unmapped = [label for label in label_names if label not in mapping]

    if unmapped and not allow_unmapped:
        raise ValueError(
            "Missing labels in mapping: " + ", ".join(sorted(unmapped))
        )

    return unmapped


def map_labels_to_groups(label_ids, id2label, label_group_mapping, *, drop_unmapped=False):
    groups = []

    for label_id in label_ids:
        label_name = id2label[label_id]
        group = label_group_mapping.get(label_name)

        if group is None:
            if drop_unmapped:
                continue
            raise ValueError(f"Label '{label_name}' not found in group mapping")

        groups.append(group)

    # dedupe for multilabel rows while keeping deterministic ordering.
    return sorted(set(groups))


def add_grouped_labels_column(df, id2label, *, scheme=None, label_group_mapping=None, input_col="labels", output_col="grouped_labels", drop_unmapped=False):
    if label_group_mapping is None:
        if scheme is None:
            raise ValueError("Provide either `scheme` or `label_group_mapping`.")
        label_group_mapping = get_label_group_mapping(scheme)

    # work on a copy to avoid mutating caller-owned dataframes in notebooks.
    df = df.copy()
    df[output_col] = df[input_col].apply(
        lambda label_ids: map_labels_to_groups(
            label_ids,
            id2label=id2label,
            label_group_mapping=label_group_mapping,
            drop_unmapped=drop_unmapped,
        )
    )
    return df
