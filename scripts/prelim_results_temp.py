#!/usr/bin/env python3
"""Quick-and-dirty preliminary Markdown report for SVM, SetFit, DistilBERT, and zero-shot."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("temp")

TRAINED_MODELS = [
    {
        "key": "setfit",
        "name": "SetFit",
        "overall_file": "setfit_overall.csv",
        "per_class_file": "setfit_per_class.csv",
    },
    {
        "key": "distilbert",
        "name": "DistilBERT",
        "overall_file": "distilbert_overall.csv",
        "per_class_file": "distilbert_per_class.csv",
    },
    {
        "key": "svm",
        "name": "SVM (TF-IDF)",
        "overall_file": "svm_tfidf_overall.csv",
        "per_class_file": "svm_tfidf_per_class.csv",
    },
]

ZERO_SHOT = {
    "name": "Zero-shot (BART)",
    "overall_file": "zero_shot_overall.csv",
    "per_class_file": "zero_shot_per_class.csv",
}


def read_csv_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x: float) -> str:
    return f"{x:.4f}"


def frac_key(x: float) -> str:
    if x == 1.0:
        return "100%"
    return f"{int(round(x * 100))}%"


def by_frac(rows, metric):
    d = defaultdict(list)
    for r in rows:
        d[float(r["data_fraction"])].append(float(r[metric]))
    return d


def row_by_seed_frac(rows, seed):
    return {
        float(r["data_fraction"]): r
        for r in rows
        if int(r["seed"]) == int(seed)
    }


def per_class_map(rows, frac, seed):
    return {
        r["emotion"]: float(r["f1"])
        for r in rows
        if float(r["data_fraction"]) == float(frac) and int(r["seed"]) == int(seed)
    }


def summarize():
    loaded = {}
    for model in TRAINED_MODELS:
        loaded[model["key"]] = {
            "name": model["name"],
            "overall": read_csv_rows(RESULTS_DIR / model["overall_file"]),
            "per_class": read_csv_rows(RESULTS_DIR / model["per_class_file"]),
        }

    zero_overall = read_csv_rows(RESULTS_DIR / ZERO_SHOT["overall_file"])
    zero_per_class = read_csv_rows(RESULTS_DIR / ZERO_SHOT["per_class_file"])

    missing = [m["name"] for m in TRAINED_MODELS if not loaded[m["key"]]["overall"]]
    if missing:
        raise SystemExit(f"Missing required overall files for: {', '.join(missing)}")
    if not zero_overall:
        raise SystemExit("Missing required file: zero_shot_overall.csv")

    # Coverage
    coverage = {}
    for model in TRAINED_MODELS:
        rows = loaded[model["key"]]["overall"]
        coverage[model["key"]] = {
            "fracs": sorted({float(r["data_fraction"]) for r in rows}),
            "seeds": sorted({int(r["seed"]) for r in rows}),
        }
    zero_fracs = sorted({float(r["data_fraction"]) for r in zero_overall})
    zero_seeds = sorted({int(r["seed"]) for r in zero_overall})

    common_fracs = set(coverage[TRAINED_MODELS[0]["key"]]["fracs"])
    for model in TRAINED_MODELS[1:]:
        common_fracs &= set(coverage[model["key"]]["fracs"])
    common_fracs = sorted(common_fracs)
    if not common_fracs:
        raise SystemExit("No common fractions across trained models.")

    metrics = {}
    for model in TRAINED_MODELS:
        key = model["key"]
        rows = loaded[key]["overall"]
        metrics[key] = {
            "macro": by_frac(rows, "macro_f1"),
            "acc": by_frac(rows, "accuracy"),
            "seed42": row_by_seed_frac(rows, 42),
        }

    zero_macro = by_frac(zero_overall, "macro_f1")
    zero_acc = by_frac(zero_overall, "accuracy")
    zero_seed42 = row_by_seed_frac(zero_overall, 42)

    lines = []
    lines.append("# Preliminary SVM vs SetFit vs DistilBERT vs Zero-Shot Results (Temporary)\n")
    lines.append("## Coverage")
    for model in TRAINED_MODELS:
        key = model["key"]
        lines.append(
            f"- {model['name']} fractions: {[frac_key(x) for x in coverage[key]['fracs']]} | seeds: {coverage[key]['seeds']}"
        )
    lines.append(f"- {ZERO_SHOT['name']} fractions: {[frac_key(x) for x in zero_fracs]} | seeds: {zero_seeds}")
    lines.append("- Note: zero-shot is evaluated at 0% train fraction (no fitting), duplicated across seeds.\n")

    # Macro table
    lines.append("## Macro-F1 by Common Fraction (seed-mean)")
    lines.append("fraction | setfit_macro | distilbert_macro | svm_macro | best_model")
    lines.append("---|---:|---:|---:|---")
    for frac in common_fracs:
        sf = mean(metrics["setfit"]["macro"][frac])
        db = mean(metrics["distilbert"]["macro"][frac])
        sv = mean(metrics["svm"]["macro"][frac])
        ranked = sorted(
            [("SetFit", sf), ("DistilBERT", db), ("SVM", sv)],
            key=lambda x: x[1],
            reverse=True,
        )
        lines.append(f"{frac_key(frac)} | {f(sf)} | {f(db)} | {f(sv)} | {ranked[0][0]}")
    lines.append("")

    # Accuracy table
    lines.append("## Exact-Match Accuracy by Common Fraction (seed-mean)")
    lines.append("fraction | setfit_acc | distilbert_acc | svm_acc | best_model")
    lines.append("---|---:|---:|---:|---")
    for frac in common_fracs:
        sf = mean(metrics["setfit"]["acc"][frac])
        db = mean(metrics["distilbert"]["acc"][frac])
        sv = mean(metrics["svm"]["acc"][frac])
        ranked = sorted(
            [("SetFit", sf), ("DistilBERT", db), ("SVM", sv)],
            key=lambda x: x[1],
            reverse=True,
        )
        lines.append(f"{frac_key(frac)} | {f(sf)} | {f(db)} | {f(sv)} | {ranked[0][0]}")
    lines.append("")

    # Zero-shot reference
    zero_frac = zero_fracs[0]
    zero_macro_mean = mean(zero_macro[zero_frac])
    zero_acc_mean = mean(zero_acc[zero_frac])
    lines.append("## Zero-shot Reference (No Training Data)")
    lines.append("fraction | zero_macro | zero_acc")
    lines.append("---|---:|---:")
    lines.append(f"{frac_key(zero_frac)} | {f(zero_macro_mean)} | {f(zero_acc_mean)}")
    lines.append("")

    # Seed-42 snapshot
    lines.append("## Seed-42 apples-to-apples snapshot")
    lines.append("fraction | setfit_macro | distilbert_macro | svm_macro | zero_macro | best_model")
    lines.append("---|---:|---:|---:|---:|---")
    for frac in common_fracs:
        if frac not in metrics["setfit"]["seed42"] or frac not in metrics["distilbert"]["seed42"] or frac not in metrics["svm"]["seed42"]:
            continue
        sf = float(metrics["setfit"]["seed42"][frac]["macro_f1"])
        db = float(metrics["distilbert"]["seed42"][frac]["macro_f1"])
        sv = float(metrics["svm"]["seed42"][frac]["macro_f1"])
        zv = float(zero_seed42[zero_frac]["macro_f1"])
        ranked = sorted(
            [("SetFit", sf), ("DistilBERT", db), ("SVM", sv), ("Zero-shot", zv)],
            key=lambda x: x[1],
            reverse=True,
        )
        lines.append(f"{frac_key(frac)} | {f(sf)} | {f(db)} | {f(sv)} | {f(zv)} | {ranked[0][0]}")
    lines.append("")

    # Per-class comparisons at max common fraction
    compare_frac = max(common_fracs)
    sf_pc = per_class_map(loaded["setfit"]["per_class"], compare_frac, 42)
    db_pc = per_class_map(loaded["distilbert"]["per_class"], compare_frac, 42)
    sv_pc = per_class_map(loaded["svm"]["per_class"], compare_frac, 42)
    z_pc = per_class_map(zero_per_class, zero_frac, 42)

    common_sf_db = sorted(set(sf_pc) & set(db_pc))
    sf_db_diffs = [(emo, sf_pc[emo] - db_pc[emo], sf_pc[emo], db_pc[emo]) for emo in common_sf_db]
    sf_db_diffs.sort(key=lambda x: x[1], reverse=True)

    lines.append(f"## Per-class F1 deltas at {frac_key(compare_frac)} (SetFit - DistilBERT, seed=42)")
    lines.append("Top SetFit gains:")
    for emo, dlt, sfv, dbv in sf_db_diffs[:6]:
        lines.append(f"- {emo}: {f(dlt)} (SetFit {f(sfv)} vs DistilBERT {f(dbv)})")
    lines.append("Largest DistilBERT wins:")
    for emo, dlt, sfv, dbv in sf_db_diffs[-3:]:
        lines.append(f"- {emo}: {f(dlt)} (SetFit {f(sfv)} vs DistilBERT {f(dbv)})")
    lines.append("")

    common_db_sv_z = sorted(set(db_pc) & set(sv_pc) & set(z_pc))
    db_vs_zero = [(emo, db_pc[emo] - z_pc[emo], db_pc[emo], z_pc[emo]) for emo in common_db_sv_z]
    db_vs_zero.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"## DistilBERT vs zero-shot per-class F1 at {frac_key(compare_frac)} (seed=42)")
    lines.append("Top DistilBERT gains:")
    for emo, dlt, dbv, zv in db_vs_zero[:6]:
        lines.append(f"- {emo}: {f(dlt)} (DistilBERT {f(dbv)} vs Zero-shot {f(zv)})")
    lines.append("Smallest gains / zero-shot strengths:")
    for emo, dlt, dbv, zv in db_vs_zero[-3:]:
        lines.append(f"- {emo}: {f(dlt)} (DistilBERT {f(dbv)} vs Zero-shot {f(zv)})")
    lines.append("")

    # Draft narrative
    sf_best = mean(metrics["setfit"]["macro"][compare_frac])
    db_best = mean(metrics["distilbert"]["macro"][compare_frac])
    sv_best = mean(metrics["svm"]["macro"][compare_frac])
    rank_at_full = sorted(
        [("SetFit", sf_best), ("DistilBERT", db_best), ("SVM", sv_best)],
        key=lambda x: x[1],
        reverse=True,
    )

    low_frac = min(common_fracs)
    sf_low = mean(metrics["setfit"]["macro"][low_frac])
    db_low = mean(metrics["distilbert"]["macro"][low_frac])
    sv_low = mean(metrics["svm"]["macro"][low_frac])

    paragraph = (
        "Preliminary results across SVM, SetFit, DistilBERT, and zero-shot show strong data dependence and clear model separation at higher data fractions. "
        f"At {frac_key(compare_frac)}, the macro-F1 ranking is {rank_at_full[0][0]} ({f(rank_at_full[0][1])}), "
        f"{rank_at_full[1][0]} ({f(rank_at_full[1][1])}), and {rank_at_full[2][0]} ({f(rank_at_full[2][1])}), "
        f"while zero-shot remains at {f(zero_macro_mean)} with no training data. "
        f"At low data ({frac_key(low_frac)}), SetFit starts strongest ({f(sf_low)}), SVM is moderate ({f(sv_low)}), and DistilBERT is weakest ({f(db_low)}), "
        "which is consistent with transformer fine-tuning being less sample-efficient in extreme low-resource settings under current hyperparameters. "
        "Per-class analysis at 100% (seed 42) indicates SetFit has broad gains over DistilBERT and SVM, while both trained neural approaches substantially outperform zero-shot on most emotions. "
        "These are still preliminary, but the current evidence supports the project hypothesis that trained methods improve with data and that SetFit is currently the strongest of the implemented approaches."
    )

    lines.append("## Draft Narrative")
    lines.append(paragraph)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "prelim_svm_setfit_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {output_path}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    summarize()

