# Predicting Subjective Readiness in Sports Participants with Personalized Baselines and Explainable AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: PMData](https://img.shields.io/badge/Dataset-Simula%20PMData-green.svg)](https://datasets.simula.no/pmdata/)

> **Research Paper & Experimental Codebase**  
> **Authors:** Minh-Triet Nguyen* (*Corresponding Author), Phuc Thien Trinh, Hung Nguyen Trong, Thu Le  
> **Affiliation:** Department of Artificial Intelligence, FPT University, Ho Chi Minh City, Viet Nam  
> **Author GitHub:** [trietnguyen-code](https://github.com/trietnguyen-code?tab=repositories)  
> **Contact:** `trietnguyen0407@gmail.com` (Corresponding Author)  
> **Repository:** [trietnguyen-code/ADY201m](https://github.com/trietnguyen-code/ADY201m)

---

## 📌 Project Overview

This repository provides the official implementation, SQL analytical queries, and evaluation benchmarks for the research paper: **"Predicting Subjective Readiness in Sports Participants with Personalized Baselines and Explainable AI"**.

Forecasting daily subjective athletic readiness ($y_{i,t+1} \in [0, 10]$) provides coaches and sports practitioners with actionable lead time to proactively adjust training loads, mitigate overreaching, and reduce injury risk. Using the open-access **PMData** lifelog collection (16 recreationally active adult participants tracked over 152 observation days, totaling 2,178 recorded participant-days), this project establishes a leakage-aware analytical pipeline integrating continuous **Fitbit Versa 2 telemetry**, daily **PMSys mobile wellness questionnaires**, and **Session-RPE exertion logs** under strict causal alignment.

```
Multi-Modal Streams (Fitbit + PMSys + sRPE) 
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Multi-Modal Ingestion & Calendar Synchronization  │
│  - Continuous 24h calendar grid (2,178 participant-days)    │
│  - Strictly past-only causal history (X_<=t)                │
│  - 1,324 valid consecutive daily transition pairs (t -> t+1)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 & 3: Causal Feature Engineering (22 Predictors)    │
│  - Subjective wellness (fatigue, soreness, sleep quality)   │
│  - Expanding personal baselines (ready_base, ready_dev)     │
│  - Wearable biometrics (resting HR, active min, sleep stages│
│  - Training workload (sRPE, ACWR acute:chronic ratio)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Three-Track Methodological Evaluation Framework   │
│  ├─ Track 1: Chronological 70/30 Split Benchmarking (10 ML) │
│  ├─ Track 2: Explainability Faithfulness (5 Explainers)     │
│  └─ Track 3: 16-Fold Leave-One-Person-Out (LOPO) Transfer   │
└─────────────────────────────────────────────────────────────┘
```

### 🎯 Key Research Contributions

1. **Leakage-Aware Causal Forecasting (Track 1):** Benchmarks ten supervised machine learning algorithms across six distinct model families against persistence on chronological 70/30 within-subject splits (920 training transitions, 404 temporal holdout test transitions). **Extra Trees** achieves the lowest error ($\text{MAE} = 0.982$, $\text{RMSE} = 1.283$, $R^2 = 0.400$), reducing prediction error by **9.25%** over within-subject persistence ($1.082$). A 1,000-sample participant-clustered bootstrap confirms Extra Trees superiority in 98.2% of resamples (95% CI $[0.807, 1.166]$).
2. **Empirical XAI Faithfulness via Progressive Deletion (Track 2):** Evaluates attribution reliability across five post-hoc explainer families (*TreeSHAP, KernelSHAP, LIME, Permutation Importance, Tree Gain*) using progressive feature neutralization without retraining. Top-5 feature deletion inflates error by $\approx 35\%$ ($\text{MAE} = 1.310 - 1.355$), substantially outpacing random deletion ($1.133$, $+15\%$). Global TreeSHAP reveals that personal baselines ($33.1\%$ top-ten mass) and wearable biometrics ($33.1\%$ across five modalities) serve as balanced predictive anchors.
3. **Cross-Subject Generalizability & The Tree Shrinkage Barrier (Track 3):** Under 16-fold Leave-One-Person-Out (LOPO) cross-validation across all 1,324 transitions, expanding baselines reduce generic error from $1.395 \to 1.247$, improving predictions for **13 of 16 participants** ($p = 0.021$, two-tailed binomial sign test).
4. **Subgroup Reversal & Pre-Deployment Calibration:** Tree ensembles pull zero-shot predictions toward the cohort mean ($4.91$), causing high error on extreme set-point outliers (e.g., p05 with $\mu=2.78$, p13 with $\mu=7.36$). While cohort persistence appears superior across all 16 subjects due to outlier distortion, excluding the three boundary cases (p03, p05, p13) reverses the ranking: on the remaining **13 regular athletic participants, personalized ML achieves superior average accuracy over persistence ($1.171$ vs. $1.212$)**. This justifies a pre-deployment screening heuristic for practical coaching systems.

---

## 📥 Dataset Setup Guide

> ⚠️ **Data Notice:**  
> The raw PMData archive is large (~10 GB uncompressed, containing minute-level JSON telemetry, sleep logs, and food images). It is not hosted directly on GitHub and is excluded via `.gitignore`. You can download the dataset automatically via Notebook A or manually as described below.

* **Official Source:** [Simula Research Laboratory PMData](https://datasets.simula.no/pmdata/)
* **Direct Zip Download:** [https://datasets.simula.no/downloads/pmdata.zip](https://datasets.simula.no/downloads/pmdata.zip)

### Option A: Automatic Download via Notebook A (Recommended)
Running **Step 1a** in [`Notebook_A_PMData_ReadinessForecast.ipynb`](Notebook_A_PMData_ReadinessForecast.ipynb) automatically downloads `pmdata.zip` and uncompresses it directly into `data/raw/`:
```python
FILES = [('https://datasets.simula.no/downloads/pmdata.zip', 'pmdata.zip')]
MARKER = 'data/raw/**/p[0-9][0-9]/fitbit/steps.json'
if not glob.glob(MARKER, recursive=True):
    for url, name in FILES: fetch(url, name)
    unzip_all()
```

### Option B: Manual Download & Extraction
1. Download `pmdata.zip` from [https://datasets.simula.no/downloads/pmdata.zip](https://datasets.simula.no/downloads/pmdata.zip).
2. Create the folder `data/raw/` in the project root directory.
3. Extract `pmdata.zip` such that participant subdirectories (`p01` through `p16`) reside in `data/raw/`:

```text
PMData_ReadinessForecast/
├── data/
│   ├── raw/
│   │   ├── p01/
│   │   │   ├── fitbit/
│   │   │   │   ├── steps.json
│   │   │   │   ├── resting_heart_rate.json
│   │   │   │   ├── sleep.json
│   │   │   │   ├── sedentary_minutes.json
│   │   │   │   └── very_active_minutes.json
│   │   │   └── pmsys/
│   │   │       ├── wellness.csv
│   │   │       └── srpe.csv
│   │   ├── p02/ ... p16/
│   │   └── participant-overview.xlsx
│   └── processed/          <-- Auto-generated by Notebook A
│       ├── manifest.json
│       ├── pmdata_daily.parquet
│       └── pmdata_hourly.parquet
```

---

## 🛠️ Installation & Environment Setup

**Prerequisites:** Python 3.10 or higher.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/trietnguyen-code/ADY201m.git
   cd ADY201m
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Key libraries:* `duckdb`, `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `lightgbm`, `xgboost`, `shap`, `scipy`, `matplotlib`, `nbformat`, `nbclient`.

---

## 🚀 Execution & Reproducibility Workflow

The pipeline is organized into modular, hash-verified stages to guarantee strict end-to-end reproducibility:

### Step 1: Data Ingestion, Cleaning & SQL Feature Engineering
Launch and execute:
```bash
jupyter notebook Notebook_A_PMData_ReadinessForecast.ipynb
```
* **Tasks performed:** Parses minute-level Fitbit JSON records and PMSys logs, performs continuous daily calendar alignment, generates `pmdata_daily.parquet` and `pmdata_hourly.parquet`, executes DuckDB exploratory analyses, and writes cryptographic MD5 checksums into `data/processed/manifest.json`.

### Step 2: Interactive SQL Analysis via DuckDB
Launch and review:
```bash
jupyter notebook sql/queries_output.ipynb
```
* **Tasks performed:** Executes the complete suite of **9 analytical SQL queries** (`Q1` to `Q9`) directly against the processed Parquet files using embedded DuckDB. Pre-rendered results, previews, and scientific domain interpretations are displayed, with CSV exports saved in `sql/queries_results/`.

### Step 3: Model Training, XAI Faithfulness & LOPO Evaluation
Launch and execute:
```bash
jupyter notebook Notebook_B_PMData_ReadinessForecast.ipynb
```
* **Tasks performed:** Verifies data integrity from `manifest.json`, trains 10 machine learning models across 5 random seeds (0–4), executes progressive feature neutralization across 5 post-hoc explainers, performs 16-fold LOPO cross-validation, and exports all LaTeX tables and CSV metrics into `report/`.

### Step 4: Reproducing Camera-Ready Publication Figures
Generate the vector PDF publication figures (`fig_pipeline.pdf`, `fig_shap.pdf`, `fig_athlete_profiles.pdf`):
```bash
python "figures code/figures.py" --data-dir report --output-dir report/figures
```

---

## 📊 Benchmark Results

### 1. Multi-Model Predictive Accuracy (Within-Subject Chronological 70/30 Holdout)
Evaluating ten supervised machine learning algorithms across six distinct model families on 404 temporal holdout transitions across all 16 participants:

| Model Family | Estimator | MAE $\downarrow$ | RMSE $\downarrow$ | $R^2$ $\uparrow$ | MAE Seed SD $\downarrow$ | Fit Time (s) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Randomized Ensemble** | **Extra Trees** | **0.982** | **1.283** | **0.400** | 0.0016 | 0.586 |
| Randomized Ensemble | Random Forest | 1.018 | 1.319 | 0.366 | 0.0032 | 0.774 |
| Regularized Linear | Elastic Net | 1.049 | 1.345 | 0.340 | 0.0000 | 0.003 |
| Regularized Linear | Ridge | 1.056 | 1.351 | 0.334 | 0.0000 | 0.003 |
| Gradient Boosted Tree | XGBoost | 1.071 | 1.383 | 0.302 | 0.0076 | 2.087 |
| Instance-based | $k$-NN | 1.073 | 1.405 | 0.280 | 0.0000 | 0.002 |
| *Baseline Heuristic* | *Persistence ($\widehat{y}_{t+1} = y_t$)* | *1.082* | *1.578* | *0.092* | *0.0000* | *0.000* |
| Gradient Boosted Tree | LightGBM | 1.114 | 1.427 | 0.258 | 0.0080 | 1.326 |
| Gradient Boosted Tree | HistGB | 1.139 | 1.462 | 0.221 | 0.0000 | 2.892 |
| Single Tree | Decision Tree (CART) | 1.215 | 1.554 | 0.120 | 0.0000 | 0.006 |
| Neural Network | Small MLP | 1.221 | 1.848 | -0.247 | 0.0222 | 0.651 |

*Note on Selection Bias:* Because Extra Trees was selected among ten algorithms on the chronological holdout, the 9.25% reduction represents an empirical upper bound subject to holdout selection. A 1,000-sample participant-clustered bootstrap yields a 95% CI of $[0.807, 1.166]$ for Extra Trees MAE and an error reduction over Random Forest of $+0.035$ ($95\%$ clustered CI: $[0.002, 0.072]$).

---

### 2. XAI Faithfulness Benchmark (Progressive Feature Deletion)
Evaluating five post-hoc explainers on Extra Trees by sequentially neutralizing the top $k \in \{1, \dots, 20\}$ features with training-set medians without model retraining:

| Explainer Framework | MAE after Top-5 $\uparrow$ | Deletion Area $\uparrow$ | Rank Stability (Spearman) $\uparrow$ | Runtime (s) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| *Full Model (No Deletion)* | *0.982* | *n/a* | *n/a* | *n/a* |
| **Tree Gain** | 1.310 | 1.285 | **0.880** | **0.002** |
| **KernelSHAP** (50 background) | 1.353 | 1.313 | 0.736 | 8.624 |
| **LIME** (1,000 perturbations) | **1.355** | 1.310 | 0.809 | 0.744 |
| **Permutation Importance** | 1.330 | **1.322** | 0.774 | 0.909 |
| **TreeSHAP** | 1.318 | 1.307 | 0.808 | 1.231 |
| *Random Deletion (Baseline)* | *1.133* | *1.158* | *n/a* | *n/a* |

*Attribution Breakdown:* Global TreeSHAP reveals that personal expanding baseline readiness accounts for **33.1%** of top-ten attribution mass (mean $|\text{SHAP}| = 0.599$), wearable biometric telemetry accounts for **33.1%** (0.600 across nocturnal RHR, sedentary minutes, daily steps, RHR deviation, 7-day sleep), training load accounts for **15.0%** (0.272), and subjective wellness features total **51.9%**.

---

### 3. Leave-One-Person-Out (LOPO) Transfer & Subgroup Reversal
Evaluating cross-subject zero-shot generalization across all 16 held-out participants (1,324 transitions):

| Evaluation Cohort | $N$ | Mean Transitions | Cohort $\mu$ | Cohort $\sigma$ | Mean Adh. | MAE Persistence | MAE Generic | MAE Personalized | MAE Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Cohort** | **16** | 83 | 5.03 | 1.42 | 69% | **1.108** | 1.395 | 1.247 | **+0.147** ($p=0.021$) |
| **Athletic Subgroup** | **13** | 90 | 5.03 | 1.35 | 72% | 1.212 | 1.333 | **1.171** | **+0.162** |
| Pooled Transitions | 1,324 | 1,324 | 4.91 | – | – | 1.126 | 1.421 | 1.259 | +0.162 |

*Key Diagnostic Insights:*
* **Tree Shrinkage toward Cohort Mean ($4.91$):** Decision trees perform piecewise-constant regression and pull predictions toward the cohort mean, severely penalizing boundary participants (e.g., p05 with $\mu = 2.78$, signed bias $+2.34$; p13 with $\mu = 7.36$, signed bias $-2.18$; Pearson correlation with cohort distance $r = 0.744, p = 0.001$).
* **Outlier Distortion on Persistence:** Participant p03 exhibited invariant reporting ($\sigma = 0.18$, 30 transitions), yielding $\text{MAE} = 0.000$ for persistence. Excluding p03 raises cohort persistence error to $1.182$ (vs. $1.313$ for personalized ML).
* **Subgroup Performance Reversal:** When pre-deployment calibration screens out the three boundary participants (p03, p05, p13), **Personalized ML achieves lower error than persistence (1.171 vs. 1.212)** on all remaining 13 regular athletic participants.

---

## 📂 Repository Structure

```text
PMData_ReadinessForecast/
├── data/                                    # Data storage (git-ignored, see setup instructions)
│   ├── raw/                                 # Raw PMData download (p01 - p16 folders)
│   └── processed/                           # Cleaned daily & hourly Parquet files + manifest.json
│
├── figures code/
│   └── figures.py                           # Standalone publication figure generator (vector PDF)
│
├── springer paper/                          # Research paper manuscript (Springer LNCS format, 12 pages)
│   ├── main_springer.tex                    # LaTeX source in llncs document class
│   ├── references.bib                       # Standardized bibliography (27 peer-reviewed references)
│   ├── README.md                            # Directory documentation
│   └── figures/                             # Vector PDF publication figures
│
├── sql/
│   ├── queries.sql                          # Complete SQL source for 9 DuckDB analytical queries
│   ├── queries_output.ipynb                 # Interactive notebook executing Q1-Q9 with interpretations
│   └── queries_results/                     # Exported CSV tables (Q1.csv - Q9.csv)
│
├── report/
│   ├── figures/                             # High-resolution PDF figures
│   ├── tables/                              # Generated LaTeX tables (models, explainers, lopo)
│   ├── table_athlete_profiles.csv           # Detailed per-participant LOPO performance log
│   ├── table_rq3_deletion_curve.csv         # Stepwise feature neutralization data
│   ├── table_shap_values.csv                # Complete SHAP attribution matrix
│   └── params_best.json                     # Optimal hyperparameters per estimator
│
├── Notebook_A_PMData_ReadinessForecast.ipynb # Pipeline Stage A: Ingestion, cleaning & SQL analysis
├── Notebook_B_PMData_ReadinessForecast.ipynb # Pipeline Stage B: ML modeling, XAI & LOPO validation
├── requirements.txt                         # Python package specifications
└── README.md                                # Project documentation (this file)
```

---

## 🗄️ Analytical SQL Queries (DuckDB)

The project leverages embedded **DuckDB** to execute complex temporal analytics directly on Parquet files without external DBMS installation:

| Query ID | Research Domain | Analytical Technique & SQL Clauses | Primary Insight / Deliverable |
| :---: | :--- | :--- | :--- |
| **Q1** | RQ1/RQ3 | `GROUP BY`, `AVG`, `STDDEV`, `ORDER BY` | Baseline readiness heterogeneity across participants ($\mu \in [2.78, 7.36]$), justifying personal baseline tracking. (`Q1.csv`) |
| **Q2** | RQ1 | CTE, `WINDOW (LEAD)`, `CASE`, `HAVING` | Next-morning readiness and sleep following hard ($sRPE \ge 400$) vs. rest days. Identifies athlete workout self-selection. (`Q2.csv`) |
| **Q3** | RQ1 | `HOUR()`, `DAYOFWEEK()`, `GROUP BY` | Diurnal step activity rhythms: weekday commute peaks (06:00-08:00) vs. weekend delayed midday sustained activity (10:00-14:00). (`Q3.csv`) |
| **Q4** | Pipeline | Multi-statement `CREATE TABLE`, `ROWS BETWEEN PRECEDING` | Validates causal feature table `feat`. Confirms exactly 1,324 valid contiguous transition pairs without lookahead leakage. (`Q4.csv`) |
| **Q5** | RQ1 | Multi-column `COUNT`, `ROUND`, ratio calculation | Contrasts passive sensor compliance (steps: 88-100%) against subjective logging adherence (PMSys: 76.6% mean, 23.4% missing). (`Q5.csv`) |
| **Q6** | RQ1/RQ2 | CTE, `WINDOW (AVG PRECEDING)`, `CASE` | Bins Acute:Chronic Workload Ratio (ACWR). Workload spikes ($>1.5$) elevate resting heart rate by $+0.22$ bpm above baseline. (`Q6.csv`) |
| **Q7** | RQ1/RQ2 | CTE, multi-tier `CASE`, interaction analysis | Disentangles sleep duration from perceived sleep quality. High quality preserves readiness even during short sleep ($<6$h). (`Q7.csv`) |
| **Q8** | RQ1 | `DAYNAME()`, `DAYOFWEEK()`, temporal aggregation | Uncovers weekly microcycles: lowest readiness on weekends (4.77-4.80) despite longer sleep, peaking on Monday (5.15) post-rest. (`Q8.csv`) |
| **Q9** | Validation | CTE, `LEAD()`, `DATE_DIFF()`, continuity bins | Verifies 24h daily transition pairing ($t \to t+1$); confirms contiguity and absence of unadjusted multi-day gaps. (`Q9.csv`) |

---

## 📖 Citation

If you use this codebase, methodology, or results in your research, please cite:

```bibtex
@article{nguyen2026predicting,
  author  = {Nguyen, Minh-Triet and Trinh, Phuc Thien and Nguyen Trong, Hung and Le, Thu},
  title   = {Predicting Subjective Readiness in Sports Participants with Personalized Baselines and Explainable AI},
  journal = {Research Manuscript / Technical Report},
  year    = {2026}
}
```

Please also cite the underlying PMData benchmark dataset:

```bibtex
@inproceedings{thambawita2020pmdata,
  author    = {Thambawita, Vajira and Hicks, Steven A. and Borgli, Hanna and Stensland, H{\aa}kon K. and Jha, Debesh and Svensen, Martin K. and Pettersen, Svein-Arne and Johansen, Dag and Johansen, H{\aa}vard D. and Pettersen, Vilde D. and others},
  title     = {{PMData}: A Sports Logging Dataset},
  booktitle = {Proceedings of the 11th ACM Multimedia Systems Conference (MMSys '20)},
  pages     = {231--236},
  year      = {2020},
  doi       = {10.1145/3339825.3394926}
}
```

---

## 📄 License & Acknowledgments

* **License:** This project is licensed under the [MIT License](LICENSE).
* **Acknowledgments:** This research was supported by **FPT University, Ho Chi Minh City, Viet Nam**. We express our sincere gratitude to the **Simula Research Laboratory** (Norway) for curating and making the PMData lifelog collection openly available for scientific research.

---

## ✉️ Contact & Support

For questions, academic inquiries, or bug reports, please open a GitHub Issue or contact the authors:
* **Minh-Triet Nguyen (Corresponding Author):** `trietnguyen0407@gmail.com` | [GitHub Profile](https://github.com/trietnguyen-code?tab=repositories)
* **Thu Le:** `thulvm@fpt.edu.vn`
* **Department of Artificial Intelligence, FPT University, Ho Chi Minh City, Viet Nam**
