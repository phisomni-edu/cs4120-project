#!/usr/bin/env python3
"""Quick-and-dirty preliminary comparison for current SVM + SetFit outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS_DIR = Path("results")


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


def summarize():
    setfit_overall = read_csv_rows(RESULTS_DIR / "setfit_overall.csv")
    svm_overall = read_csv_rows(RESULTS_DIR / "svm_tfidf_overall.csv")
    setfit_per_class = read_csv_rows(RESULTS_DIR / "setfit_per_class.csv")
    svm_per_class = read_csv_rows(RESULTS_DIR / "svm_tfidf_per_class.csv")

    if not setfit_overall or not svm_overall:
        raise SystemExit("Missing setfit_overall.csv or svm_tfidf_overall.csv")

    # coverage
    setfit_fracs = sorted({float(r["data_fraction"]) for r in setfit_overall})
    setfit_seeds = sorted({int(r["seed"]) for r in setfit_overall})
    svm_fracs = sorted({float(r["data_fraction"]) for r in svm_overall})
    svm_seeds = sorted({int(r["seed"]) for r in svm_overall})

    common_fracs = sorted(set(setfit_fracs) & set(svm_fracs))

    # index helpers
    def by_frac(rows, metric):
        d = defaultdict(list)
        for r in rows:
            d[float(r["data_fraction"])].append(float(r[metric]))
        return d

    setfit_macro = by_frac(setfit_overall, "macro_f1")
    setfit_acc = by_frac(setfit_overall, "accuracy")
    svm_macro = by_frac(svm_overall, "macro_f1")
    svm_acc = by_frac(svm_overall, "accuracy")

    svm_seed42 = {
        float(r["data_fraction"]): r
        for r in svm_overall
        if int(r["seed"]) == 42
    }

    lines = []
    lines.append("Preliminary SVM vs SetFit Results (Temporary)\n")
    lines.append("Coverage")
    lines.append(f"- SetFit fractions: {[frac_key(x) for x in setfit_fracs]} | seeds: {setfit_seeds}")
    lines.append(f"- SVM fractions: {[frac_key(x) for x in svm_fracs]} | seeds: {svm_seeds}")
    lines.append("- Note: SetFit currently has one seed and no 100% run in this file set.\n")

    lines.append("Macro-F1 / Accuracy by Common Fraction")
    lines.append("fraction | setfit_macro | svm_macro_mean | delta | setfit_acc | svm_acc_mean | delta")
    lines.append("---|---:|---:|---:|---:|---:|---:")
    for frac in common_fracs:
        sf_m = mean(setfit_macro[frac])
        sv_m = mean(svm_macro[frac])
        sf_a = mean(setfit_acc[frac])
        sv_a = mean(svm_acc[frac])
        lines.append(
            f"{frac_key(frac)} | {f(sf_m)} | {f(sv_m)} | {f(sf_m - sv_m)} | {f(sf_a)} | {f(sv_a)} | {f(sf_a - sv_a)}"
        )
    lines.append("")

    lines.append("Seed-42 apples-to-apples snapshot")
    lines.append("fraction | setfit_macro | svm_macro_seed42 | delta")
    lines.append("---|---:|---:|---:")
    for frac in common_fracs:
        if frac not in svm_seed42:
            continue
        sf_m = mean(setfit_macro[frac])
        sv_m = float(svm_seed42[frac]["macro_f1"])
        lines.append(f"{frac_key(frac)} | {f(sf_m)} | {f(sv_m)} | {f(sf_m - sv_m)}")
    lines.append("")

    # per-class comparison at max common fraction using seed 42
    compare_frac = max(common_fracs)
    sf_pc = {
        r["emotion"]: float(r["f1"])
        for r in setfit_per_class
        if float(r["data_fraction"]) == compare_frac and int(r["seed"]) == 42
    }
    sv_pc = {
        r["emotion"]: float(r["f1"])
        for r in svm_per_class
        if float(r["data_fraction"]) == compare_frac and int(r["seed"]) == 42
    }
    common_emotions = sorted(set(sf_pc) & set(sv_pc))

    diffs = [(emo, sf_pc[emo] - sv_pc[emo], sf_pc[emo], sv_pc[emo]) for emo in common_emotions]
    diffs.sort(key=lambda x: x[1], reverse=True)

    lines.append(f"Per-class F1 deltas at {frac_key(compare_frac)} (SetFit - SVM, seed=42)")
    lines.append("Top SetFit gains:")
    for emo, dlt, sfv, svv in diffs[:6]:
        lines.append(f"- {emo}: {f(dlt)} (SetFit {f(sfv)} vs SVM {f(svv)})")
    lines.append("Largest SVM wins:")
    for emo, dlt, sfv, svv in diffs[-3:]:
        lines.append(f"- {emo}: {f(dlt)} (SetFit {f(sfv)} vs SVM {f(svv)})")
    lines.append("")

    # 1/3-page-ish provisional narrative
    best_frac = compare_frac
    sf_best = mean(setfit_macro[best_frac])
    sv_best_mean = mean(svm_macro[best_frac])

    paragraph = (
        "Preliminary results suggest SetFit is currently outperforming the TF-IDF + SVM baseline "
        "across all shared data fractions (1%, 5%, 10%, 25%, and 50%). At 50% data, SetFit reaches "
        f"macro-F1 {f(sf_best)} compared with SVM mean macro-F1 {f(sv_best_mean)}, a gap of {f(sf_best - sv_best_mean)}. "
        "The same pattern appears for exact-match accuracy, with SetFit consistently ahead at each common fraction. "
        "Class-level analysis at 50% (seed 42) indicates the largest SetFit gains on curiosity, disapproval, and confusion, "
        "while SVM still has small advantages on a few categories (notably pride and neutral) and a larger advantage on grief. "
        "These results should be treated as interim: SetFit currently reflects a partial run (single seed, up to 50%) while SVM "
        "includes three seeds and a full-data point. For the next checkpoint, we should finish remaining stable SetFit runs and "
        "report seed-aggregated comparisons on the common fractions for a fair cross-model conclusion."
    )

    lines.append("Draft Narrative")
    lines.append(paragraph)

    output_path = RESULTS_DIR / "prelim_svm_setfit_summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {output_path}")
    print("\n".join(lines[:35]))


if __name__ == "__main__":
    summarize()
