
# DAQUA: A Data Quality, Stability, Leakage, and Model-Ranking Readiness Framework for Reliable Software Defect Prediction

This repository contains the implementation and experimental scripts for the research paper:

**DAQUA: A Data Quality, Stability, Leakage, and Model-Ranking Readiness Framework for Reliable Software Defect Prediction**

DAQUA is a pre-claim, model-assisted audit framework for software defect prediction datasets. Instead of asking only *which classifier performs best*, DAQUA asks whether a dataset family provides enough evidence to support reliable prediction, cross-project transfer, model comparison, and explanation.

---

## 1. Main Objective

Software defect prediction studies often compare learning algorithms on benchmark datasets and assume that the datasets are reliable enough for empirical conclusions. DAQUA makes this assumption explicit by auditing each dataset family before strong predictive or comparative claims are made.

DAQUA evaluates each dataset family through six readiness dimensions:

| Symbol | Dimension | Purpose |
|---|---|---|
| `Q` | Data-quality readiness | Measures structural data quality, cleaning effects, imbalance, outliers, redundancy, and related risks. |
| `C` | Data-complexity readiness | Measures intrinsic prediction difficulty, class overlap, neighborhood consistency, and borderline instances. |
| `S` | Cross-project stability | Measures whether projects inside the same dataset family are comparable for cross-project or transfer claims. |
| `L` | Leakage-signal readiness | Detects suspicious features, temporal/post-release signals, duplicate contamination, and possible leakage risks. |
| `M` | Model-ranking stability | Measures whether model rankings remain stable across projects, metrics, and repeated evaluations. |
| `E` | Explanation readiness | Measures whether feature-importance explanations are stable across runs, projects, and explanation methods. |

The final DAQUA score is computed as:

```text
DAQUA = 0.20Q + 0.18C + 0.18S + 0.17L + 0.12M + 0.15E
````

The output is not just one score. DAQUA produces:

* a six-dimensional readiness profile,
* an overall DAQUA readiness score,
* warnings for detected risks,
* evidence-coverage information,
* risk-aware evaluation recommendations,
* robustness checks for scoring, model ranking, and explanation readiness.

---

## 2. Repository Structure

```text
DAQUA/
├── daqua/
│   ├── loaders/
│   │   └── defect_loader.py
│   ├── profiling/
│   │   ├── quality.py
│   │   ├── complexity.py
│   │   ├── stability.py
│   │   ├── leakage.py
│   │   └── readiness.py
│   ├── models/
│   │   └── baseline_models.py
│   ├── explainability/
│   │   ├── feature_importance.py
│   │   └── explanation_stability.py
│   ├── reports/
│   │   └── report_generator.py
│   └── main.py
│
├── scripts/
│   ├── run_full_daqua.sh
│   ├── preprocess_bughunter_file_level.py
│   ├── daqua_scoring_sensitivity.py
│   ├── daqua_model_ranking_robustness.py
│   └── daqua_explanation_readiness_robustness.py
│
├── Data-set/
│   └── <software defect datasets>
│
├── outputs/
│   ├── profiles_9datasets_full/
│   ├── reports_9datasets_full/
│   └── robustness_9datasets/
│
└── README.md
```

---

## 3. Dataset Families

The final DAQUA evaluation uses nine software defect dataset families comprising 51 project-level datasets.

| Dataset family         | Projects | Granularity                     |
| ---------------------- | -------: | ------------------------------- |
| AEEEM                  |        5 | Class level                     |
| ApacheJIT              |        1 | Commit level                    |
| BugHunter              |       15 | Source-code entity / file level |
| JIRA                   |        7 | Release / module level          |
| Kamei                  |        3 | Commit level                    |
| NASA MDP               |        9 | Module level                    |
| PROMISE                |        7 | Module / class level            |
| ReLink                 |        3 | File level                      |
| UnifiedBugDataSet File |        1 | File level                      |
| **Total**              |   **51** | Mixed granularity               |

Two additional datasets, `ContinuousDefect` and `GHPR`, are skipped in the current implementation because they do not expose a direct supervised binary defect label in the available tabular representation.

---

## 4. Environment Setup

Create and activate a Python environment:

```bash
conda create -n daqua python=3.9 -y
conda activate daqua
```

Install dependencies:

```bash
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib tqdm liac-arff
```

Or, if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

---

## 5. Dataset Placement

Place all dataset folders inside:

```text
Data-set/
```

Expected example layout:

```text
Data-set/
├── AEEEM/
├── ApacheJIT/
├── BugHunter/
├── JIRA/
├── Kamei/
├── NASA/
├── Promise/
├── ReLink/
├── UnifiedBugDataSet_File/
├── ContinuousDefect/
└── GHPR_dataset-master/
```

The loader automatically detects supported defect-label columns such as:

```text
bugs, bug, #bugs, Number of Bugs, NumberOfBugs,
Defective, isDefective, RealBug, buggy, defects, label, class
```

Non-zero bug counts are converted to the defective class.

---

## 6. Running the Full DAQUA Pipeline

From the repository root:

```bash
cd ~/Home/SPARC
```

Run the full DAQUA audit:

```bash
bash scripts/run_full_daqua.sh
```

The expected full-run output folders are:

```text
outputs/profiles_9datasets_full/
outputs/reports_9datasets_full/
outputs/logs/
```

The main report is generated at:

```text
outputs/reports_9datasets_full/DAQUA_summary_report_9datasets_full.md
```

The main profile files include:

```text
outputs/profiles_9datasets_full/DAQUA_quality_summary_by_dataset.csv
outputs/profiles_9datasets_full/DAQUA_complexity_summary_by_dataset.csv
outputs/profiles_9datasets_full/DAQUA_stability_summary_by_dataset.csv
outputs/profiles_9datasets_full/DAQUA_leakage_summary_by_dataset.csv
outputs/profiles_9datasets_full/DAQUA_model_ranking_stability.csv
outputs/profiles_9datasets_full/DAQUA_explanation_readiness.csv
outputs/profiles_9datasets_full/DAQUA_readiness_scores.csv
```

---

## 7. Running Robustness Analyses

The repository includes three robustness analyses used for the research paper.

---

### 7.1 Scoring Sensitivity Analysis

This script evaluates whether DAQUA conclusions change under alternative scoring configurations:

* equal dimension weights,
* equal indicator weights,
* ±20% dimension-weight perturbation,
* ±20% indicator-weight perturbation,
* ±20% threshold perturbation,
* logistic normalization,
* quantile normalization,
* ±5 readiness-boundary shifts.

Run:

```bash
cd ~/Home/SPARC

mkdir -p outputs/robustness_9datasets/scoring_sensitivity

python3 scripts/daqua_scoring_sensitivity.py \
  2>&1 | tee outputs/robustness_9datasets/scoring_sensitivity/run.log
```

Outputs:

```text
outputs/robustness_9datasets/scoring_sensitivity/
├── all_configuration_scores.csv
├── dimension_weights.csv
├── indicator_specification_table.csv
├── per_configuration_comparison.csv
├── readiness_boundaries.csv
├── sensitivity_summary.csv
├── tab_sensitivity.tex
└── warning_thresholds_and_severity.csv
```

---

### 7.2 Model-Ranking Robustness

This script evaluates whether model-ranking stability changes when:

* AUC rankings are compared with MCC rankings,
* one classifier is removed from the classifier pool.

Run:

```bash
cd ~/Home/SPARC

rm -rf outputs/robustness_9datasets/model_ranking
mkdir -p outputs/robustness_9datasets/model_ranking

python3 scripts/daqua_model_ranking_robustness.py \
  2>&1 | tee outputs/robustness_9datasets/model_ranking/run.log
```

Outputs:

```text
outputs/robustness_9datasets/model_ranking/
├── combined_auc_mcc_stability.csv
├── leave_one_classifier_out.csv
├── leave_one_classifier_out_summary.csv
├── metric_conclusion_comparison.csv
├── metric_ranking_stability.csv
├── project_model_metric_values.csv
├── tab_model_metric_robustness.tex
├── tab_model_loo_robustness.tex
└── run.log
```

Note: PR-AUC robustness is marked as unavailable in the current run because the saved baseline outputs contain aggregate AUC and MCC values, but not project-level prediction probabilities.

---

### 7.3 Explanation-Readiness Robustness

This script compares explanation readiness under two feature-importance methods:

* native Random Forest feature importance,
* permutation importance.

Run the resource-safe final configuration:

```bash
cd ~/Home/SPARC

rm -rf outputs/robustness_9datasets/explanation_readiness_final
mkdir -p outputs/robustness_9datasets/explanation_readiness_final

python3 -u scripts/daqua_explanation_readiness_robustness.py \
  --root Data-set \
  --profiles-dir outputs/profiles_9datasets_full \
  --out-dir outputs/robustness_9datasets/explanation_readiness_final \
  --n-splits 5 \
  --n-repeats 1 \
  --n-estimators 80 \
  --n-permutation-repeats 1 \
  --max-permutation-rows 1000 \
  --random-state 42 \
  2>&1 | tee outputs/robustness_9datasets/explanation_readiness_final/run.log
```

Outputs:

```text
outputs/robustness_9datasets/explanation_readiness_final/
├── cross_project_explanation_stability_by_method.csv
├── explanation_method_comparison.csv
├── explanation_method_global_summary.csv
├── explanation_readiness_by_method.csv
├── feature_importance_native_vs_permutation.csv
├── project_aggregated_feature_profiles_by_method.csv
├── project_explanation_stability_by_method.csv
├── project_native_permutation_top10_overlap.csv
├── tab_explanation_method_robustness.tex
└── run.log
```

---

## 8. Main DAQUA Results

The final DAQUA evaluation ranks the nine dataset families as follows.

| Rank | Dataset                | Quality | Complexity | Stability | Leakage | Model stability | Explanation readiness | DAQUA | Readiness                     |
| ---: | ---------------------- | ------: | ---------: | --------: | ------: | --------------: | --------------------: | ----: | ----------------------------- |
|    1 | UnifiedBugDataSet File |   74.78 |      66.03 |     80.00 |  100.00 |         100.00* |                 60.07 | 79.25 | Ready with reporting controls |
|    2 | Kamei                  |   89.71 |      57.57 |     79.88 |   90.00 |           96.35 |                 56.82 | 78.07 | Ready with reporting controls |
|    3 | ReLink                 |   82.17 |      65.73 |     76.19 |  100.00 |           80.08 |                 48.80 | 75.91 | Ready with reporting controls |
|    4 | ApacheJIT              |   84.70 |      60.83 |     80.00 |   90.00 |         100.00* |                 39.04 | 75.45 | Ready with reporting controls |
|    5 | AEEEM                  |   78.47 |      66.03 |     71.77 |   79.28 |           86.10 |                 63.73 | 73.87 | Usable with caution           |
|    6 | JIRA                   |   77.57 |      62.87 |     83.59 |   89.88 |           84.25 |                 39.22 | 73.15 | Usable with caution           |
|    7 | BugHunter              |   66.68 |      40.72 |     68.91 |   97.70 |           83.84 |                 78.66 | 71.54 | Usable with caution           |
|    8 | PROMISE                |   81.65 |      60.84 |     60.03 |   96.58 |           75.02 |                 51.89 | 71.29 | Usable with caution           |
|    9 | NASA MDP               |   77.01 |      58.36 |     73.49 |   95.34 |           79.33 |                 35.50 | 70.19 | Usable with caution           |

* ApacheJIT and UnifiedBugDataSet File are single-project families in the current configuration. Their cross-project model-ranking stability is not empirically estimable and should be interpreted as provisional.



## 9. Key Findings

1. No evaluated dataset family provides uniformly strong evidence across all six readiness dimensions.
2. Dataset families with similar DAQUA scores often have different limiting risks.
3. PROMISE appears strong in data quality and leakage readiness but remains limited for cross-project stability and explanation readiness.
4. BugHunter has the weakest complexity readiness but the strongest default explanation-readiness profile.
5. Kamei has the strongest data-quality and model-ranking stability profile.
6. NASA MDP has the lowest overall DAQUA readiness score in the final evaluation.
7. Scoring conclusions are stable under reasonable weighting perturbations.
8. Model-ranking conclusions are metric- and classifier-pool-dependent.
9. Explanation-readiness conclusions are strongly dependent on the feature-importance method.

---

## 10. Generated Output Files

After running the full pipeline and robustness scripts, the main files are:

```text
outputs/profiles_9datasets_full/DAQUA_readiness_scores.csv
outputs/reports_9datasets_full/DAQUA_summary_report_9datasets_full.md

outputs/robustness_9datasets/scoring_sensitivity/tab_sensitivity.tex
outputs/robustness_9datasets/model_ranking/tab_model_metric_robustness.tex
outputs/robustness_9datasets/model_ranking/tab_model_loo_robustness.tex
outputs/robustness_9datasets/explanation_readiness_final/tab_explanation_method_robustness.tex
```

---

## 11. Reproducibility Notes

* All reported results are conditional on the documented DAQUA audit configuration.
* ApacheJIT and UnifiedBugDataSet File are single-project families; cross-project stability and model-ranking agreement are therefore not fully estimable for them.
* Leakage indicators are audit signals, not definitive proof of leakage.
* PR-AUC robustness is unavailable in the current aggregate metric output because prediction probabilities were not stored.
* Explanation-readiness robustness uses a resource-safe configuration with 5 folds, 1 repeat, 80 Random Forest estimators, and 1 permutation repeat.

---

## 12. Contact

For questions related to the implementation or paper, please contact:

```text
Abhinav Jamwal
Department of Computer Science and Engineering
Indian Institute of Technology Roorkee
```

MD

````

After creating it, check:

```bash
head -n 40 README.md
````

Then commit it:

```bash
git add README.md
git commit -m "Add detailed DAQUA README"
```

This README is appropriate for GitHub because it includes both: the **repository objective** and a **compact paper-result summary with tables**.
