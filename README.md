# CS4120 Project: Few-Shot Emotion Detection

Comparing few-shot, zero-shot, and supervised approaches to multi-class emotion classification on the GoEmotions dataset.

---

## Repo Structure

cs4120-project/    
├── README.md    
├── requirements.txt    
├── data/    
│   └── README.md    
├── notebooks/    
│   ├── 00_eda.ipynb                 # Person 1 — EDA, preprocessing, subsampling    
│   ├── 01_svm_baseline.ipynb        # Person 4 — TF-IDF + SVM baseline    
│   ├── 02_distilbert.ipynb          # Person 2 — DistilBERT fine-tuning    
│   ├── 03_setfit.ipynb              # Person 3 — SetFit few-shot experiments    
│   ├── 04_zero_shot.ipynb           # Person 4 — Zero-shot classification    
│   └── 05_analysis_viz.ipynb        # Person 4 — Analysis and visualizations    
├── src/    
│   ├── data_utils.py                # Shared subsampling and preprocessing utilities    
│   ├── evaluate.py                  # Shared evaluation framework    
│   └── label_mapping.py             # Emotion grouping logic    
└── results/    
    └── README.md    

---

## Getting Started

All work is done in Google Colab. You do not need to install anything locally.

### Step 1 — Open your notebook in Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → GitHub tab
3. Paste the repo URL and select your notebook

### Step 2 — Set the correct runtime
Each notebook has a different runtime requirement — check the instructions at the top of your notebook before running anything.

### Step 3 — Run the setup cell
The first cell in every notebook handles installs, Drive mounting, and imports. Run it before anything else. When prompted, authorize Google Drive access.

### Step 4 — Connect Colab to GitHub
The first time you save (Ctrl+S), Colab will ask you to authorize GitHub access. Accept this — it allows Colab to save your notebook directly to the repo.

---

## Workflow

### Starting a session
1. Open your notebook in Colab (Step 1 above)
2. Set the correct runtime
3. Run the setup cell
4. You're ready to work

### Saving your work
There are three types of things to save:

**Notebooks** — Ctrl+S saves the notebook directly to GitHub. Do this regularly as you work.

**Results and figures** (CSVs, plots) — save these to the `results/` folder in the repo:
1. Your code saves the file to `/content/results/` during the session
2. Right-click the file in the Colab left sidebar → Download
3. Go to the repo on GitHub → `results/` → Add file → Upload files → Commit

**Model checkpoints** (large files) — these go to Google Drive, not GitHub:
```python
# Each notebook has the correct save call for its model — see notebook instructions
model.save_pretrained(SAVE_DIR + "checkpoints/your_model")
```

---

## Data

The dataset is loaded directly from Hugging Face — no manual download needed:

```python
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
```

**Dependency:** All notebooks except `00_eda.ipynb` load preprocessed data from the `data/` folder in this repo. Person 1 must commit the processed CSVs before anyone else can run their experiments. Check that the following files exist in `data/` before starting:

```
data/train.csv
data/val.csv
data/test.csv
data/train_1pct.csv
data/train_5pct.csv
data/train_10pct.csv
data/train_25pct.csv
data/train_50pct.csv
```

Once they're there, load them like this:
```python
import pandas as pd
train_df = pd.read_csv('data/train_10pct.csv')
test_df = pd.read_csv('data/test.csv')
```

---

## Results Format

To make sure all results can be combined for the final analysis, save your results CSV in this format:

| method | data_fraction | seed | emotion | f1 | precision | recall |
|---|---|---|---|---|---|---|
| setfit | 0.01 | 42 | joy | 0.72 | 0.74 | 0.70 |

```python
results_df.to_csv('results/{your_method}_results.csv', index=False)
```

---

## Dependencies

All dependencies are installed automatically in each notebook's setup cell. No local installation required.

Key libraries: `transformers`, `datasets`, `setfit`, `accelerate`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
