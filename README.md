# NF1 Tumour-Status Classification Using a Contextual Bandit and Local Phenotype Evidence

This project performs concurrent classification of tumour status from co-recorded clinical phenotype features in neurofibromatosis type 1 (NF1). It does not address NF1 diagnosis or prospective risk prediction. The data comes from the public [UCI NF1 clinical phenotype resource](https://archive.ics.uci.edu/dataset/1162/neurofibromatosis+type+1+clinical+symptoms+of+familial+and+sporadic+cases). The experiments use a locally derived working table of 296 records with 17 predictors, which is not included in this repository. The script expects it as `dataset_csv.csv`.

- Data preprocessing: The target is `Tumour Case`. Case Type, identifier columns, and the Healthy indicator are excluded from the inputs. Missing values are imputed with an iterative imputer and features are scaled with a robust scaler, both fitted only on the training partition of each fold.
- Modeling approach: A contextual bandit agent with two linear action-value models and epsilon-greedy exploration is trained with reward-driven updates. A local phenotype score is computed from the 15 nearest training neighbours with inverse-distance weighting. The two scores are combined by a fixed weighted fusion (0.7 agent, 0.3 local).
- Algorithms used: Contextual bandit (SGD regressors), k-nearest-neighbour local evidence, and fixed weighted score fusion.
- Evaluation: Repeated stratified cross-validation with 5 folds and 10 repeats. Patient-level aggregated scores are evaluated with ROC-AUC, PR-AUC, F1-score, MCC, and other metrics, and the script saves the tables and figures reported in the paper.

## Paper

Arman Ahangari Kiasari and Ali Ghasemi Sahebi, "Concurrent Tumour-Status Classification in Neurofibromatosis Type 1 Using Contextual Bandit and Local Phenotype Evidence," 12th International Conference on Signal Processing and Intelligent Systems (ICSPIS), IEEE, 2026.
