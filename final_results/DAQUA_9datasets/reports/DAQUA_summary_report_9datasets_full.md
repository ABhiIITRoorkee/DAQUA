# DAQUA Summary Report

## Overview

DAQUA evaluates whether software defect datasets are reliable, stable, leakage-aware, model-ranking-stable, explanation-stability-aware, and suitable for trustworthy prediction before model comparison.

- Number of dataset families: **9**
- Number of projects: **51**

## Readiness Ranking

| dataset | quality_score | complexity_score | stability_score | leakage_score | model_stability_score | explanation_readiness_score | overall_readiness_score | readiness_level | prediction_readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UnifiedBugDataSet_File | 74.78 | 66.03 | 80.00 | 100.00 | 100.00 | 60.07 | 79.25 | moderate | ready_with_reporting_controls |
| Kamei | 89.71 | 57.57 | 79.88 | 90.00 | 96.35 | 56.82 | 78.07 | moderate | ready_with_reporting_controls |
| ReLink | 82.17 | 65.73 | 76.19 | 100.00 | 80.08 | 48.80 | 75.91 | moderate | ready_with_reporting_controls |
| ApacheJIT | 84.70 | 60.83 | 80.00 | 90.00 | 100.00 | 39.04 | 75.45 | moderate | ready_with_reporting_controls |
| AEEEM | 78.47 | 66.03 | 71.77 | 79.28 | 86.10 | 63.73 | 73.87 | moderate | usable_with_caution |
| JIRA | 77.57 | 62.87 | 83.59 | 89.88 | 84.25 | 39.22 | 73.15 | moderate | usable_with_caution |
| BugHunter | 66.68 | 40.72 | 68.91 | 97.70 | 83.84 | 78.66 | 71.54 | moderate | usable_with_caution |
| Promise | 81.65 | 60.84 | 60.03 | 96.58 | 75.02 | 51.89 | 71.29 | moderate | usable_with_caution |
| NASA | 77.01 | 58.36 | 73.49 | 95.34 | 79.33 | 35.50 | 70.19 | moderate | usable_with_caution |


## Quality Summary

| dataset | n_projects | row_retention_rate | minority_class_percentage | class_imbalance_ratio | high_correlation_feature_rate | outlier_instance_rate | quality_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AEEEM | 5 | 0.86 | 0.22 | 4.98 | 0.29 | 0.38 | 78.47 |
| ApacheJIT | 1 | 1.00 | 0.27 | 2.77 | 0.40 | 0.75 | 84.70 |
| BugHunter | 15 | 0.88 | 0.41 | 1.66 | 0.27 | 0.27 | 66.68 |
| JIRA | 7 | 0.96 | 0.11 | 11.94 | 0.42 | 0.56 | 77.57 |
| Kamei | 3 | 1.00 | 0.31 | 2.33 | 0.04 | 0.55 | 89.71 |
| NASA | 9 | 0.81 | 0.16 | 10.19 | 0.57 | 0.44 | 77.01 |
| Promise | 7 | 0.86 | 0.17 | 23.97 | 0.13 | 0.47 | 81.65 |
| ReLink | 3 | 0.96 | 0.40 | 1.60 | 0.76 | 0.36 | 82.17 |
| UnifiedBugDataSet_File | 1 | 0.82 | 0.28 | 2.59 | 0.00 | 0.32 | 74.78 |


## Complexity Summary

| dataset | n_projects | minority_class_percentage | nn_same_class_ratio | minority_nn_same_class_ratio | borderline_instance_rate | mean_feature_overlap | complexity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AEEEM | 5 | 0.22 | 0.81 | 0.48 | 0.16 | 0.44 | 66.03 |
| ApacheJIT | 1 | 0.27 | 0.76 | 0.54 | 0.19 | 0.89 | 60.83 |
| BugHunter | 15 | 0.41 | 0.38 | 0.26 | 0.47 | 0.78 | 40.72 |
| JIRA | 7 | 0.11 | 0.88 | 0.36 | 0.10 | 0.55 | 62.87 |
| Kamei | 3 | 0.31 | 0.68 | 0.46 | 0.29 | 0.80 | 57.57 |
| NASA | 9 | 0.16 | 0.81 | 0.32 | 0.16 | 0.63 | 58.36 |
| Promise | 7 | 0.17 | 0.82 | 0.39 | 0.15 | 0.56 | 60.84 |
| ReLink | 3 | 0.40 | 0.67 | 0.61 | 0.32 | 0.67 | 65.73 |
| UnifiedBugDataSet_File | 1 | 0.28 | 0.66 | 0.40 | 0.27 | 0.36 | 66.03 |


## Stability Summary

| dataset | n_projects | project_size_cv | feature_set_consistency | label_prevalence_range | mean_pairwise_ks | mean_pairwise_stability_score | stability_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AEEEM | 5 | 0.54 | 1.00 | 0.40 | 0.23 | 80.49 | 71.77 |
| ApacheJIT | 1 | 0.00 | 1.00 | 0.00 | NA | NA | 80.00 |
| BugHunter | 15 | 1.61 | 1.00 | 0.36 | 0.22 | 82.25 | 68.91 |
| JIRA | 7 | 0.31 | 1.00 | 0.22 | 0.16 | 88.74 | 83.59 |
| Kamei | 3 | 0.76 | 1.00 | 0.12 | 0.33 | 79.79 | 79.88 |
| NASA | 9 | 0.73 | 0.96 | 0.33 | 0.22 | 82.22 | 73.49 |
| Promise | 7 | 1.85 | 1.00 | 0.97 | 0.20 | 75.29 | 60.03 |
| ReLink | 3 | 0.60 | 1.00 | 0.20 | 0.37 | 76.57 | 76.19 |
| UnifiedBugDataSet_File | 1 | 0.00 | 1.00 | 0.00 | NA | NA | 80.00 |


## Leakage Summary

| dataset | n_projects | mean_suspicious_feature_rate | mean_temporal_feature_count | mean_post_release_feature_count | mean_high_label_correlation_count | mean_cross_project_duplicate_rate | mean_leakage_score | leakage_level | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AEEEM | 5 | 0.26 | 0.00 | 1.00 | 0.00 | 0.00 | 79.28 | moderate_leakage_risk | possible_cross_project_contamination;possible_post_release_features;suspicious_feature_names |
| ApacheJIT | 1 | 0.07 | 1.00 | 1.00 | 0.00 | 0.00 | 90.00 | low_leakage_risk | possible_post_release_features;suspicious_feature_names |
| BugHunter | 15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 97.70 | low_leakage_risk | possible_cross_project_contamination |
| JIRA | 7 | 0.13 | 3.00 | 0.00 | 0.00 | 0.00 | 89.88 | moderate_leakage_risk | suspicious_feature_names |
| Kamei | 3 | 0.07 | 0.00 | 1.00 | 0.00 | 0.00 | 90.00 | low_leakage_risk | possible_post_release_features;suspicious_feature_names |
| NASA | 9 | 0.00 | 2.00 | 0.00 | 0.00 | 0.05 | 95.34 | low_leakage_risk | possible_cross_project_contamination |
| Promise | 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 96.58 | low_leakage_risk | possible_cross_project_contamination |
| ReLink | 3 | 0.00 | 3.00 | 0.00 | 0.00 | 0.00 | 100.00 | low_leakage_risk | none |
| UnifiedBugDataSet_File | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | low_leakage_risk | none |


## Model-Ranking Stability Summary

| dataset | n_projects | n_models | mean_pairwise_spearman_auc | mean_pairwise_kendall_auc | mean_pairwise_spearman_mcc | mean_pairwise_kendall_mcc | model_ranking_stability_score | model_ranking_stability_level | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AEEEM | 5 | 6 | 0.90 | 0.81 | 0.65 | 0.52 | 86.10 | high | none |
| ApacheJIT | 1 | 6 | NA | NA | NA | NA | NA | not_available | insufficient_projects_for_ranking_stability |
| BugHunter | 15 | 6 | 0.71 | 0.60 | 0.75 | 0.64 | 83.84 | moderate | none |
| JIRA | 7 | 6 | 0.89 | 0.81 | 0.58 | 0.47 | 84.25 | moderate | none |
| Kamei | 3 | 6 | 1.00 | 1.00 | 0.89 | 0.82 | 96.35 | high | none |
| NASA | 9 | 6 | 0.81 | 0.71 | 0.45 | 0.37 | 79.33 | moderate | none |
| Promise | 7 | 6 | 0.76 | 0.63 | 0.34 | 0.28 | 75.02 | moderate | none |
| ReLink | 3 | 6 | 0.85 | 0.73 | 0.45 | 0.38 | 80.08 | moderate | none |
| UnifiedBugDataSet_File | 1 | 6 | NA | NA | NA | NA | NA | not_available | insufficient_projects_for_ranking_stability |


## Explanation Readiness Summary

| dataset | n_projects | mean_project_feature_importance_stability | mean_pairwise_project_top10_jaccard | mean_pairwise_project_spearman | dataset_feature_importance_stability_score | explanation_readiness_score | explanation_readiness_level | primary_risks | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BugHunter | 15 | 69.43 | 1.00 | 0.55 | 82.35 | 78.66 | moderate | none | none |
| AEEEM | 5 | 61.04 | 0.64 | 0.51 | 66.82 | 63.73 | limited | limited_explanation_readiness | limited_feature_importance_stability;limited_explanation_readiness |
| UnifiedBugDataSet_File | 1 | 92.42 | NA | NA | 92.42 | 60.07 | limited | limited_explanation_readiness | limited_explanation_readiness |
| Kamei | 3 | 58.12 | 0.50 | 0.33 | 58.22 | 56.82 | limited | limited_explanation_readiness;low_cross_project_feature_rank_correlation | limited_feature_importance_stability;limited_explanation_readiness;low_cross_project_feature_rank_correlation |
| Promise | 7 | 50.60 | 0.46 | 0.45 | 56.21 | 51.89 | limited | limited_explanation_readiness;low_cross_project_feature_rank_correlation | limited_feature_importance_stability;limited_explanation_readiness;low_cross_project_feature_rank_correlation |
| ReLink | 3 | 39.23 | 0.48 | 0.54 | 54.53 | 48.80 | low | low_explanation_readiness;unstable_within_project_feature_importance | limited_feature_importance_stability;low_explanation_readiness;unstable_within_project_feature_importance |
| JIRA | 7 | 34.48 | 0.18 | 0.54 | 43.25 | 39.22 | low | low_explanation_readiness;unstable_within_project_feature_importance;unstable_dataset_level_feature_importance;low_top10_explanation_overlap | unstable_feature_importance;low_explanation_readiness;unstable_within_project_feature_importance;unstable_dataset_level_feature_importance;low_top10_explanation_overlap |
| ApacheJIT | 1 | 60.06 | NA | NA | 60.06 | 39.04 | low | low_explanation_readiness | limited_feature_importance_stability;low_explanation_readiness |
| NASA | 9 | 32.86 | 0.23 | 0.31 | 40.59 | 35.50 | low | low_explanation_readiness;unstable_within_project_feature_importance;unstable_dataset_level_feature_importance;low_top10_explanation_overlap;low_cross_project_feature_rank_correlation | unstable_feature_importance;low_explanation_readiness;unstable_within_project_feature_importance;unstable_dataset_level_feature_importance;low_top10_explanation_overlap;low_cross_project_feature_rank_correlation |


## Dataset Risks and Recommendations

### UnifiedBugDataSet_File

- **Primary risks:** `limited_explanation_readiness_risk;minority_class_overlap_risk`
- **Warnings:** `insufficient_projects_for_ranking_stability;limited_explanation_readiness;limited_explanation_readiness_risk;minority_class_overlap_risk`
- **Recommended protocol:** `report_cleaning_impact;cross_project_evaluation;repeated_runs;avoid_single_threshold_claims;report_stability_sensitivity;report_explanation_stability;avoid_strong_global_explanation_claims;treat_explanations_as_dataset_conditional`
- **Recommended metrics:** `AUC;MCC;F1;G-mean;P@20;R@20;IFA;TopK_AUC;feature_importance_stability;top_k_feature_overlap;explanation_rank_correlation;explanation_variance`

### Kamei

- **Primary risks:** `prediction_complexity_risk;limited_explanation_readiness_risk;feature_distribution_shift_risk;minority_class_overlap_risk;suspicious_feature_name_risk;possible_post_release_feature_risk`
- **Warnings:** `moderate_feature_distribution_shift;possible_post_release_features;suspicious_feature_names;limited_feature_importance_stability;limited_explanation_readiness;low_cross_project_feature_rank_correlation;prediction_complexity_risk;limited_explanation_readiness_risk;feature_distribution_shift_risk;minority_class_overlap_risk;suspicious_feature_name_risk;possible_post_release_feature_risk`
- **Recommended protocol:** `cross_project_evaluation_with_shift_analysis;repeated_runs_with_project_level_reporting;avoid_single_threshold_claims;report_stability_sensitivity;audit_suspicious_feature_names;verify_feature_availability_time;report_explanation_stability;avoid_strong_global_explanation_claims;treat_explanations_as_dataset_conditional`
- **Recommended metrics:** `AUC;MCC;F1;G-mean;per_project_results;failure_rate;feature_importance_stability;top_k_feature_overlap;explanation_rank_correlation;explanation_variance`

### ReLink

- **Primary risks:** `moderate_model_ranking_stability_risk;low_explanation_readiness_risk;feature_redundancy_risk;feature_distribution_shift_risk;borderline_instance_risk`
- **Warnings:** `moderate_feature_distribution_shift;limited_feature_importance_stability;low_explanation_readiness;unstable_within_project_feature_importance;moderate_model_ranking_stability_risk;low_explanation_readiness_risk;feature_redundancy_risk;feature_distribution_shift_risk;borderline_instance_risk`
- **Recommended protocol:** `cross_project_evaluation_with_shift_analysis;repeated_runs;avoid_single_threshold_claims;report_stability_sensitivity;feature_redundancy_analysis;report_model_ranking_stability;avoid_single_metric_model_claims;report_explanation_stability;avoid_strong_global_explanation_claims;treat_explanations_as_dataset_conditional`
- **Recommended metrics:** `AUC;MCC;F1;G-mean;model_ranking_stability;metric_sensitivity_analysis;feature_importance_stability;top_k_feature_overlap;explanation_rank_correlation;explanation_variance`

### ApacheJIT

- **Primary risks:** `prediction_complexity_risk;low_explanation_readiness_risk;suspicious_feature_name_risk;possible_post_release_feature_risk`
- **Warnings:** `possible_post_release_features;suspicious_feature_names;insufficient_projects_for_ranking_stability;limited_feature_importance_stability;low_explanation_readiness;prediction_complexity_risk;low_explanation_readiness_risk;suspicious_feature_name_risk;possible_post_release_feature_risk`
- **Recommended protocol:** `cross_project_evaluation;repeated_runs_with_project_level_reporting;report_stability_sensitivity;audit_suspicious_feature_names;verify_feature_availability_time;report_explanation_stability;avoid_strong_global_explanation_claims;treat_explanations_as_dataset_conditional`
- **Recommended metrics:** `AUC;MCC;F1;G-mean;P@20;R@20;IFA;TopK_AUC;per_project_results;failure_rate;feature_importance_stability;top_k_feature_overlap;explanation_rank_correlation;explanation_variance`

### AEEEM

- **Primary risks:** `moderate_leakage_risk;limited_explanation_readiness_risk;label_distribution_shift_risk;minority_class_overlap_risk;suspicious_feature_name_risk;possible_post_release_feature_risk`
- **Warnings:** `large_label_prevalence_range;possible_cross_project_contamination;possible_post_release_features;suspicious_feature_names;limited_feature_importance_stability;limited_explanation_readiness;moderate_leakage_risk;limited_explanation_readiness_risk;label_distribution_shift_risk;minority_class_overlap_risk;suspicious_feature_name_risk;possible_post_release_feature_risk`
- **Recommended protocol:** `cross_project_evaluation_with_shift_analysis;repeated_runs;avoid_single_threshold_claims;report_stability_sensitivity;inspect_leakage_warnings;audit_suspicious_feature_names;verify_feature_availability_time;report_explanation_stability;avoid_strong_global_explanation_claims;treat_explanations_as_dataset_conditional`
- **Recommended metrics:** `AUC;MCC;F1;G-mean;P@20;R@20;IFA;TopK_AUC;confidence_intervals;run_variance;leakage_sensitivity_analysis;feature_importance_stability;top_k_feature_overlap;explanation_rank_correlation;explanation_variance`

### JIRA

- **Primary risks:** `prediction_complexity_risk;moderate_leakage_risk;moderate_model_ranking_stability_risk;low_explanation_readiness_risk;class_imbalance_risk;feature_redundancy_risk;label_distribution_shift_risk;minority_class_overlap_risk;suspicious_feature_name_risk`
- **Warnings:** `moderate_label_prevalence_range;suspicious_feature_names;unstable_feature_importance;low_explanation_readiness;unstable_within_project_feature_importance;unstable_dataset_level_feature_importance;low_top10_explanation_overlap;prediction_complexity_risk;moderate_leakage_risk;moderate_model_ranking_stability_risk;low_explanation_readiness_risk;class_imbalance_risk;feature_redundancy_risk;label_distribution_shift_risk;minority_class_overlap_risk;suspicious_feature_name_risk`
- **Recommended protocol:** `cross_project_evaluation_with_shift_analysis;repeated_runs_with_project_level_reporting;imbalance_aware_metrics;avoid_single_threshold_claims;report_stability_sensitivity;inspect_leakage_warnings;audit_suspicious_feature_names;feature_redundancy_analysis;report_model_ranking_stability;avoid_single_metric_model_claims;report_explanation_stability;avoid_strong_global_explanation_claims;treat_explanations_as_dataset_conditional`
- **Recommended metrics:** `AUC;MCC;F1;G-mean;P@20;R@20;IFA;TopK_AUC;per_project_results;failure_rate;leakage_sensitivity_analysis;model_ranking_stability;metric_sensitivity_analysis;feature_importance_stability;top_k_feature_overlap;explanation_rank_correlation;explanation_variance`

### BugHunter

- **Primary risks:** `data_quality_risk;prediction_complexity_risk;dataset_stability_risk;moderate_model_ranking_stability_risk;label_distribution_shift_risk;minority_class_overlap_risk;borderline_instance_risk;project_size_instability_risk;possible_cross_project_contamination_risk`
- **Warnings:** `unstable_project_sizes;large_label_prevalence_range;possible_cross_project_contamination;data_quality_risk;prediction_complexity_risk;dataset_stability_risk;moderate_model_ranking_stability_risk;label_distribution_shift_risk;minority_class_overlap_risk;borderline_instance_risk;project_size_instability_risk;possible_cross_project_contamination_risk`
- **Recommended protocol:** `report_cleaning_impact;cross_project_evaluation_with_shift_analysis;repeated_runs_with_project_level_reporting;avoid_single_threshold_claims;avoid_strong_generalization_claims;deduplicate_across_projects_before_transfer;report_model_ranking_stability;avoid_single_metric_model_claims`
- **Recommended metrics:** `AUC;MCC;F1;G-mean;confidence_intervals;run_variance;per_project_results;failure_rate;model_ranking_stability;metric_sensitivity_analysis`

### Promise

- **Primary risks:** `prediction_complexity_risk;dataset_stability_risk;moderate_model_ranking_stability_risk;limited_explanation_readiness_risk;class_imbalance_risk;label_distribution_shift_risk;minority_class_overlap_risk;project_size_instability_risk;possible_cross_project_contamination_risk`
- **Warnings:** `unstable_project_sizes;large_label_prevalence_range;possible_cross_project_contamination;limited_feature_importance_stability;limited_explanation_readiness;low_cross_project_feature_rank_correlation;prediction_complexity_risk;dataset_stability_risk;moderate_model_ranking_stability_risk;limited_explanation_readiness_risk;class_imbalance_risk;label_distribution_shift_risk;minority_class_overlap_risk;project_size_instability_risk;possible_cross_project_contamination_risk`
- **Recommended protocol:** `cross_project_evaluation_with_shift_analysis;repeated_runs_with_project_level_reporting;imbalance_aware_metrics;avoid_single_threshold_claims;avoid_strong_generalization_claims;deduplicate_across_projects_before_transfer;report_model_ranking_stability;avoid_single_metric_model_claims;report_explanation_stability;avoid_strong_global_explanation_claims;treat_explanations_as_dataset_conditional`
- **Recommended metrics:** `AUC;MCC;F1;G-mean;P@20;R@20;IFA;TopK_AUC;confidence_intervals;run_variance;per_project_results;failure_rate;model_ranking_stability;metric_sensitivity_analysis;feature_importance_stability;top_k_feature_overlap;explanation_rank_correlation;explanation_variance`

### NASA

- **Primary risks:** `prediction_complexity_risk;moderate_model_ranking_stability_risk;low_explanation_readiness_risk;class_imbalance_risk;feature_redundancy_risk;label_distribution_shift_risk;minority_class_overlap_risk;possible_cross_project_contamination_risk`
- **Warnings:** `large_label_prevalence_range;possible_cross_project_contamination;unstable_feature_importance;low_explanation_readiness;unstable_within_project_feature_importance;unstable_dataset_level_feature_importance;low_top10_explanation_overlap;low_cross_project_feature_rank_correlation;prediction_complexity_risk;moderate_model_ranking_stability_risk;low_explanation_readiness_risk;class_imbalance_risk;feature_redundancy_risk;label_distribution_shift_risk;minority_class_overlap_risk;possible_cross_project_contamination_risk`
- **Recommended protocol:** `cross_project_evaluation_with_shift_analysis;repeated_runs_with_project_level_reporting;imbalance_aware_metrics;avoid_single_threshold_claims;report_stability_sensitivity;deduplicate_across_projects_before_transfer;feature_redundancy_analysis;report_model_ranking_stability;avoid_single_metric_model_claims;report_explanation_stability;avoid_strong_global_explanation_claims;treat_explanations_as_dataset_conditional`
- **Recommended metrics:** `AUC;MCC;F1;G-mean;P@20;R@20;IFA;TopK_AUC;confidence_intervals;run_variance;per_project_results;failure_rate;model_ranking_stability;metric_sensitivity_analysis;feature_importance_stability;top_k_feature_overlap;explanation_rank_correlation;explanation_variance`


## Research Interpretation

Across the analysed defect dataset families, the mean DAQUA readiness score was 74.30. The highest-ranked dataset was UnifiedBugDataSet_File with a readiness score of 79.25, while the lowest-ranked dataset was NASA with a score of 70.19. 2 dataset families showed nontrivial leakage warnings, which should be treated as audit signals rather than definitive evidence of leakage. 5 dataset families showed less than high model-ranking stability, suggesting that claims about the best model should be reported with metric sensitivity and project-level stability analysis. 8 dataset families showed limited or low explanation readiness, indicating that feature-importance explanations should be treated as dataset-conditional rather than universally stable explanations. These findings support the DAQUA argument that software defect prediction datasets should be audited before algorithmic comparison, because predictive performance, model ranking, explanation reliability, leakage sensitivity, and cross-project generalization are conditioned by measurable dataset properties.
