from typing import Dict, Iterable, List, Sequence


# GoEmotions labels grouped into Ekman's six basic emotions.
# Labels that do not cleanly map to Ekman are assigned to "other".
EKMAN_6_GROUP_MAPPING: Dict[str, str] = {
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


# Coarser sentiment-style grouping used in many class-balance analyses.
POS_NEG_AMBIGUOUS_NEUTRAL_MAPPING: Dict[str, str] = {
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


def get_label_group_mapping(scheme: str) -> Dict[str, str]:
    """Return label->group mapping for a supported grouping scheme."""
    normalized = scheme.strip().lower()

    if normalized in {"ekman", "ekman6", "ekman_6"}:
        return EKMAN_6_GROUP_MAPPING.copy()
    if normalized in {"pos_neg_ambiguous_neutral", "sentiment4"}:
        return POS_NEG_AMBIGUOUS_NEUTRAL_MAPPING.copy()

    raise ValueError(
        f"Unsupported scheme '{scheme}'. Supported schemes: {sorted(SUPPORTED_SCHEMES)}"
    )


def validate_label_group_mapping(
    label_names: Sequence[str],
    mapping: Dict[str, str],
    *,
    allow_unmapped: bool = False,
) -> List[str]:
    """Validate mapping coverage and return the list of unmapped labels."""
    unmapped = [label for label in label_names if label not in mapping]

    if unmapped and not allow_unmapped:
        raise ValueError(
            "Missing labels in mapping: " + ", ".join(sorted(unmapped))
        )

    return unmapped


def map_labels_to_groups(
    label_ids: Iterable[int],
    id2label: Dict[int, str],
    label_group_mapping: Dict[str, str],
    *,
    drop_unmapped: bool = False,
    preserve_order: bool = False,
) -> List[str]:
    """
    Map one example's label IDs to grouped label names.

    Returns unique groups. By default groups are sorted for deterministic outputs.
    """
    groups: List[str] = []

    for label_id in label_ids:
        label_name = id2label[label_id]
        group = label_group_mapping.get(label_name)

        if group is None:
            if drop_unmapped:
                continue
            raise ValueError(f"Label '{label_name}' not found in group mapping")

        groups.append(group)

    if preserve_order:
        deduped: List[str] = []
        seen = set()
        for group in groups:
            if group not in seen:
                deduped.append(group)
                seen.add(group)
        return deduped

    return sorted(set(groups))


def add_grouped_labels_column(
    df,
    id2label: Dict[int, str],
    *,
    scheme: str | None = None,
    label_group_mapping: Dict[str, str] | None = None,
    input_col: str = "labels",
    output_col: str = "grouped_labels",
    drop_unmapped: bool = False,
):
    """Add a grouped-label column to a dataframe using either a scheme or explicit mapping."""
    if label_group_mapping is None:
        if scheme is None:
            raise ValueError("Provide either `scheme` or `label_group_mapping`.")
        label_group_mapping = get_label_group_mapping(scheme)

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
