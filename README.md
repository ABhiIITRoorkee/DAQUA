# DAQUA: A Data Quality, Stability, Leakage, and Model-Ranking Readiness Framework for Reliable Software Defect Prediction

This repository contains the implementation and experimental scripts for the research paper:

**DAQUA: A Data Quality, Stability, Leakage, and Model-Ranking Readiness Framework for Reliable Software Defect Prediction**

DAQUA is a pre-claim, model-assisted audit framework for software defect prediction datasets. Instead of asking only *which classifier performs best*, DAQUA asks whether a dataset family provides enough evidence to support reliable prediction, cross-project transfer, model comparison, and explanation.

---

## 1. Main Objective

Software defect prediction studies often compare learning algorithms on benchmark datasets and assume that the datasets are reliable enough for empirical conclusions. DAQUA makes this assumption explicit by auditing each dataset family before strong predictive, transfer, comparative, or explanation claims are made.

DAQUA evaluates each dataset family through six complementary and non-interchangeable readiness dimensions:

| Symbol | Dimension | What it assesses |
|---|---|---|
| `Q` | Data-quality readiness | Structural integrity, cleaning effects, missingness, conflicting or duplicate observations, class imbalance, outliers, low-variance features, and redundancy. |
| `C` | Data-complexity readiness | Class separability, neighborhood consistency, minority-class difficulty, borderline instances, feature overlap, and related prediction difficulty. |
| `S` | Cross-project stability | Feature-schema consistency, project-size variation, defect-prevalence shift, and feature-distribution shift relevant to transfer claims. |
| `L` | Leakage-signal readiness | Suspicious features, temporal or post-release indicators, feature-label associations, duplicate overlap, and cross-project contamination signals. |
| `M` | Model-ranking stability | Agreement among project-level model rankings across metrics, repeated evaluations, and the documented classifier pool. |
| `E` | Explanation readiness | Reproducibility of feature-importance evidence across repeated runs and projects under a documented model and importance configuration. Method dependence is assessed separately through robustness analysis. |

For a family with complete evidence in all six dimensions, the score reduces to:

```text
DAQUA = 0.20Q + 0.18C + 0.18S + 0.17L + 0.12M + 0.15E
```

The general score is availability-aware:

```text
DAQUA(D_j) = sum(w_k * R_k(D_j), k in K_j) / sum(w_k, k in K_j)
Coverage(D_j) = 100 * sum(w_k * c_jk, k in K)
```

Here, `K_j` contains the estimable dimensions and `c_jk` records the availability of the prescribed evidence. Unavailable dimensions are not assigned artificial neutral, zero, or perfect values. The score must therefore be interpreted jointly with evidence coverage, warnings, and the six-dimensional profile.

DAQUA produces:

* a six-dimensional readiness profile,
* an availability-aware overall score,
* explicit evidence coverage,
* indicator-level risk warnings,
* a configuration-dependent operational status,
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

| Dataset family | Projects | Native prediction granularity |
|---|---:|---|
| AEEEM | 5 | Class level |
| ApacheJIT | 1 | Commit level |
| BugHunter | 15 | Source-code entity level |
| JIRA | 7 | Release/module level |
| Kamei | 3 | Commit level |
| NASA MDP | 9 | Module level |
| PROMISE | 7 | Module/class level |
| ReLink | 3 | File level |
| UnifiedBugDataSet File | 1 | File level |
| **Total** | **51** | Class-, module-, file-, release-, and commit-level settings |

Two additional datasets, `ContinuousDefect` and `GHPR`, are skipped by the current repository configuration because the available tabular representations do not expose a direct supervised binary defect label. This is an implementation-scope statement, not a general claim that those datasets cannot support other prediction formulations.

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

From the repository root, run the full DAQUA audit:

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

Run from the repository root:

```bash
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

Run from the repository root:

```bash
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

Run the resource-conscious configuration used for the reported robustness analysis:

```bash
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

The table reports only dataset families with complete evidence across all six readiness dimensions. Single-project families are excluded from this ranking because their cross-project stability and model-ranking evidence are unavailable.

| Rank | Dataset | Quality | Complexity | Stability | Leakage signal | Model stability | Explanation readiness | DAQUA score | Coverage | Operational status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Kamei | 89.71 | 57.57 | 79.88 | 90.00 | 96.35 | 56.82 | 78.07 | 100.00 | Ready with controls |
| 2 | ReLink | 82.17 | 65.73 | 76.19 | 100.00 | 80.08 | 48.80 | 75.91 | 100.00 | Ready with controls |
| 3 | AEEEM | 78.47 | 66.03 | 71.77 | 79.28 | 86.10 | 63.73 | 73.87 | 100.00 | Usable with caution |
| 4 | JIRA | 77.57 | 62.87 | 83.59 | 89.88 | 84.25 | 39.22 | 73.15 | 100.00 | Usable with caution |
| 5 | BugHunter | 66.68 | 40.72 | 68.91 | 97.70 | 83.84 | 78.66 | 71.54 | 100.00 | Usable with caution |
| 6 | PROMISE | 81.65 | 60.84 | 60.03 | 96.58 | 75.02 | 51.89 | 71.29 | 100.00 | Usable with caution |
| 7 | NASA MDP | 77.01 | 58.36 | 73.49 | 95.34 | 79.33 | 35.50 | 70.19 | 100.00 | Usable with caution |

No fully evaluated family receives the unrestricted `Ready` status. Kamei has the highest complete score, followed by ReLink, but both require explicit evaluation controls. ApacheJIT and UnifiedBugDataSet File remain provisional and are discussed in the reproducibility notes rather than ranked with complete profiles.

---

## 9. Key Findings

1. No fully evaluated dataset family provides uniformly strong evidence across all six readiness dimensions or receives the unrestricted `Ready` status.
2. Kamei has the highest complete DAQUA score, followed by ReLink; both still require controls tailored to their limiting dimensions.
3. Similar aggregate scores can conceal different structural, transfer, ranking, leakage, or explanation risks.
4. PROMISE appears strong in data quality and leakage-signal readiness but is limited by project heterogeneity, metric-sensitive model rankings, and explanation readiness.
5. BugHunter has the weakest complexity-readiness profile but the strongest default explanation-readiness profile among the fully evaluated families.
6. NASA MDP has the lowest complete DAQUA score in the evaluated set, driven particularly by weak explanation reproducibility.
7. ApacheJIT and UnifiedBugDataSet File remain provisional because their single-project composition does not provide complete cross-project evidence.
8. Numerical ordering is comparatively stable under reasonable weighting and threshold changes, while operational statuses near decision boundaries are more configuration-sensitive.
9. Model-ranking and explanation conclusions should be supported by metric, classifier-pool, repetition, project, and feature-importance-method robustness checks.

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

* All reported scores and statuses are conditional on the documented DAQUA audit configuration.
* ApacheJIT and UnifiedBugDataSet File are single-project families. Their cross-project stability and model-ranking agreement are unavailable, and their explanation-readiness scores use only within-project evidence.
* Availability-renormalized provisional scores must not be compared directly with complete six-dimensional profiles.
* Leakage indicators are screening signals, not definitive proof that leakage is present or absent. Feature provenance and prediction-time availability still require verification.
* PR-AUC model-ranking robustness is unavailable because project-level prediction probabilities were not retained in the stored outputs.
* The matched explanation-method robustness analysis uses 5 folds, 1 repeat, 80 Random Forest estimators, 1 permutation repeat, at most 1,000 rows for permutation analysis, and random state 42.
* The default explanation-readiness values in the main results and the matched native-versus-permutation robustness outputs answer different questions and should not be substituted for one another.

---

## 12. Citation

If you use this repository, please cite the paper:

```bibtex
@article{jamwal2026daqua,
  title   = {DAQUA: A Data Quality, Stability, Leakage, and Model-Ranking Readiness Framework for Reliable Software Defect Prediction},
  author  = {Jamwal, Abhinav and Rodriguez, Daniel and Kumar, Sandeep and Gutierrez-Martinez, Jose-Maria},
  journal = {Under Review},
  year    = {2026}
}
```

---

## 13. Contact

For questions related to the implementation or paper, please contact:

```text
Abhinav Jamwal
Department of Computer Science and Engineering
Indian Institute of Technology Roorkee
```
