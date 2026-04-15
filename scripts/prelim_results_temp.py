#!/usr/bin/env python3
"""Quick-and-dirty preliminary Markdown report for SVM + SetFit + Zero-shot outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("temp")


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
    zero_overall = read_csv_rows(RESULTS_DIR / "zero_shot_overall.csv")
    setfit_per_class = read_csv_rows(RESULTS_DIR / "setfit_per_class.csv")
    svm_per_class = read_csv_rows(RESULTS_DIR / "svm_tfidf_per_class.csv")
    zero_per_class = read_csv_rows(RESULTS_DIR / "zero_shot_per_class.csv")

    if not setfit_overall or not svm_overall:
        raise SystemExit("Missing setfit_overall.csv or svm_tfidf_overall.csv")
    if not zero_overall:
        raise SystemExit("Missing zero_shot_overall.csv")

    # coverage
    setfit_fracs = sorted({float(r["data_fraction"]) for r in setfit_overall})
    setfit_seeds = sorted({int(r["seed"]) for r in setfit_overall})
    svm_fracs = sorted({float(r["data_fraction"]) for r in svm_overall})
    svm_seeds = sorted({int(r["seed"]) for r in svm_overall})
    zero_fracs = sorted({float(r["data_fraction"]) for r in zero_overall})
    zero_seeds = sorted({int(r["seed"]) for r in zero_overall})

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
    zero_macro = by_frac(zero_overall, "macro_f1")
    zero_acc = by_frac(zero_overall, "accuracy")

    svm_seed42 = {
        float(r["data_fraction"]): r
        for r in svm_overall
        if int(r["seed"]) == 42
    }
    setfit_seed42 = {
        float(r["data_fraction"]): r
        for r in setfit_overall
        if int(r["seed"]) == 42
    }
    zero_seed42 = {
        float(r["data_fraction"]): r
        for r in zero_overall
        if int(r["seed"]) == 42
    }

    lines = []
    lines.append("# Preliminary SVM vs SetFit vs Zero-Shot Results (Temporary)\n")
    lines.append("## Coverage")
    lines.append(f"- SetFit fractions: {[frac_key(x) for x in setfit_fracs]} | seeds: {setfit_seeds}")
    lines.append(f"- SVM fractions: {[frac_key(x) for x in svm_fracs]} | seeds: {svm_seeds}")
    lines.append(f"- Zero-shot fractions: {[frac_key(x) for x in zero_fracs]} | seeds: {zero_seeds}")
    lines.append("- Note: zero-shot is evaluated at 0% train fraction (no fitting), duplicated across seeds.\n")

    lines.append("## Macro-F1 / Accuracy by Common Fraction (SetFit vs SVM)")
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

    zero_frac = zero_fracs[0]
    zero_macro_mean = mean(zero_macro[zero_frac])
    zero_acc_mean = mean(zero_acc[zero_frac])
    lines.append("## Zero-shot Reference (No Training Data)")
    lines.append("fraction | zero_macro | zero_acc")
    lines.append("---|---:|---:")
    lines.append(f"{frac_key(zero_frac)} | {f(zero_macro_mean)} | {f(zero_acc_mean)}")
    lines.append("")

    lines.append("## Seed-42 apples-to-apples snapshot")
    lines.append("fraction | setfit_macro | svm_macro_seed42 | zero_macro | setfit-vs-svm | setfit-vs-zero | svm-vs-zero")
    lines.append("---|---:|---:|---:|---:|---:|---:")
    for frac in common_fracs:
        if frac not in setfit_seed42 or frac not in svm_seed42:
            continue
        sf_m = float(setfit_seed42[frac]["macro_f1"])
        sv_m = float(svm_seed42[frac]["macro_f1"])
        zv_m = float(zero_seed42[zero_frac]["macro_f1"])
        lines.append(
            f"{frac_key(frac)} | {f(sf_m)} | {f(sv_m)} | {f(zv_m)} | {f(sf_m - sv_m)} | {f(sf_m - zv_m)} | {f(sv_m - zv_m)}"
        )
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
    z_pc = {
        r["emotion"]: float(r["f1"])
        for r in zero_per_class
        if float(r["data_fraction"]) == zero_frac and int(r["seed"]) == 42
    }

    diffs = [(emo, sf_pc[emo] - sv_pc[emo], sf_pc[emo], sv_pc[emo]) for emo in common_emotions]
    diffs.sort(key=lambda x: x[1], reverse=True)

    lines.append(f"## Per-class F1 deltas at {frac_key(compare_frac)} (SetFit - SVM, seed=42)")
    lines.append("Top SetFit gains:")
    for emo, dlt, sfv, svv in diffs[:6]:
        lines.append(f"- {emo}: {f(dlt)} (SetFit {f(sfv)} vs SVM {f(svv)})")
    lines.append("Largest SVM wins:")
    for emo, dlt, sfv, svv in diffs[-3:]:
        lines.append(f"- {emo}: {f(dlt)} (SetFit {f(sfv)} vs SVM {f(svv)})")
    lines.append("")

    common_emotions_zero = sorted(set(sf_pc) & set(z_pc) & set(sv_pc))
    zero_diffs = [(emo, sf_pc[emo] - z_pc[emo], sf_pc[emo], z_pc[emo]) for emo in common_emotions_zero]
    zero_diffs.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"## Per-class F1 deltas at {frac_key(compare_frac)} vs zero-shot reference (seed=42)")
    lines.append("Top SetFit gains over zero-shot:")
    for emo, dlt, sfv, zv in zero_diffs[:6]:
        lines.append(f"- {emo}: {f(dlt)} (SetFit {f(sfv)} vs Zero-shot {f(zv)})")
    lines.append("Smallest gains / zero-shot strengths:")
    for emo, dlt, sfv, zv in zero_diffs[-3:]:
        lines.append(f"- {emo}: {f(dlt)} (SetFit {f(sfv)} vs Zero-shot {f(zv)})")
    lines.append("")

    # 1/3-page-ish provisional narrative
    best_frac = compare_frac
    sf_best = mean(setfit_macro[best_frac])
    sv_best_mean = mean(svm_macro[best_frac])
    sf_best_seed42 = float(setfit_seed42[best_frac]["macro_f1"])
    sv_best_seed42 = float(svm_seed42[best_frac]["macro_f1"])
    z_best_seed42 = float(zero_seed42[zero_frac]["macro_f1"])

    paragraph = (
        "Preliminary results across SVM, SetFit, and zero-shot indicate that SetFit is strongest overall. "
        "Across most trained fractions, SVM is second and zero-shot is lower, though at 1% data the current zero-shot macro-F1 is above SVM. Using all available seeds, "
        f"SetFit reaches macro-F1 {f(sf_best)} at {frac_key(best_frac)}, compared with SVM mean macro-F1 {f(sv_best_mean)} "
        f"(delta {f(sf_best - sv_best_mean)}). The zero-shot reference is macro-F1 {f(zero_macro_mean)} at 0% train data, "
        "which is below both trained approaches at higher fractions. Seed-42 comparisons across common fractions show the same pattern, and "
        f"at {frac_key(best_frac)} the seed-42 gap is SetFit {f(sf_best_seed42)} vs SVM {f(sv_best_seed42)} vs zero-shot {f(z_best_seed42)}. "
        "Per-class results also show that trained models recover stronger class-level F1 on most emotions than zero-shot. "
        "These are still preliminary and should be treated as checkpoint evidence rather than final conclusions, but they support "
        "the project hypothesis that training-based methods scale better with data for fine-grained emotion classification."
    )

    lines.append("## Draft Narrative")
    lines.append(paragraph)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "prelim_svm_setfit_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {output_path}")
    print("\n".join(lines[:35]))


if __name__ == "__main__":
    summarize()
