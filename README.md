# Emotion Classification - CS4120 Final Project

GoEmotions emotion-classification project comparing four approaches across multiple training-data fractions:

- tf-idf + linear svm baseline
- distilbert fine-tuning
- setfit few-shot learning
- zero-shot bart-mnli

The repo includes:

- shared data/preprocessing utilities in `src/data_utils.py`
- shared label grouping in `src/label_mapping.py`
- shared multilabel evaluation in `src/evaluate.py`
- experiment notebooks in `notebooks/`
- result csv outputs in `results/`

## Quick Start (local):

1. clone the repo and enter it:

```bash
git clone <repo-url>
cd cs4120-project
```

2. create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # windows powershell: .venv\\Scripts\\Activate.ps1
```

3. install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. start jupyter:

```bash
jupyter lab
```

## Quick Start (Colab):

For `02_distilbert.ipynb` and `03_setfit.ipynb`, Colab GPU runtimes are recommended

1. open notebook in Colab.
2. set runtime type to GPU.
3. run cells top-to-bottom. setup cells clone the repo into `/content/cs4120-project` and install notebook-specific packages.
4. if you want checkpoint backups, mount drive when prompted in notebooks that support it.

## Notebook Run Order:

Run in this order if starting from raw data:

1. `notebooks/00_eda.ipynb`
2. `notebooks/01_svm_baseline.ipynb`
3. `notebooks/02_distilbert.ipynb`
4. `notebooks/03_setfit.ipynb`
5. `notebooks/04_zero_shot.ipynb`
6. `notebooks/05_analysis_viz.ipynb`

### What notebook 00 produces:

All downstream notebooks expect processed files in `data/`:

- `train.csv`, `val.csv`, `test.csv`
- `train_clean.csv`, `validation_clean.csv`, `test_clean.csv`
- fraction subsets: `train_1pct*.csv`, `train_5pct*.csv`, `train_10pct*.csv`, `train_25pct*.csv`, `train_50pct*.csv`

If those files already exist (as in this repo), you can skip rerunning 00.

## Running each experiment:

- `01_svm_baseline.ipynb`: cpu is fine.
- `02_distilbert.ipynb`: gpu strongly recommended.
- `03_setfit.ipynb`: gpu strongly recommended.
- `04_zero_shot.ipynb`: can run on cpu or gpu.
- `05_analysis_viz.ipynb`: reads result csvs and generates figures.

## Outputs:

Main outputs are written to `results/`:

- `svm_tfidf_overall.csv`, `svm_tfidf_per_class.csv`, `svm_tfidf_results.csv`
- `distilbert_overall.csv`, `distilbert_per_class.csv`, `distilbert_results.csv`
- `setfit_overall.csv`, `setfit_per_class.csv`, `setfit_results.csv`
- `zero_shot_overall.csv`, `zero_shot_per_class.csv`, `zero_shot_results.csv`


## Notes:

- the task is multilabel emotion classification on goemotions.
- shared metrics come from `src/evaluate.py` (accuracy, macro-f1, micro-f1, hamming loss + per-class metrics).
- experiment seeds are typically `[42, 7, 21]` and fractions `[0.01, 0.05, 0.10, 0.25, 0.50, 1.00]`.
