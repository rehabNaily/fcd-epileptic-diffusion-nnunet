# Kaggle notebooks

All computations ran as Kaggle notebooks on an NVIDIA T4 GPU. Run them in this order; each one reads the outputs of the previous ones as Kaggle inputs.

| # | Notebook | Output | In this repository |
|---|---|---|---|
| 00 | Preprocessing (`new-data`, V2.1): OpenNeuro download and integrity check, FLAIR 1 mm grid, rigid T1→FLAIR registration, HD-BET, N4, z-score, cropping, quality control | `ds004199_preproc/final/` | `00_preprocessing_new-data_V2.1_export.txt` (cells and outputs) |
| 01 | Experimental plan: subject roles and acquisition protocols | `plan_experimental.csv` | to add |
| 02 | DDPM and reference autoencoder training on healthy controls | model weights, loss logs | to add |
| 03 | Diagnosis of the diffusion model (synthetic lesions, choice of t*) | diagnostic tables and figures | to add |
| 04 | Refinement of the anomaly score (step A) | `reglages_A.json`, variant table | to add |
| 05 | Anomaly maps for all patients | `anomaly_maps/` | to add |
| 06 | nnU-Net baseline (FLAIR, T1), fold 0 | `nnunet_baseline/` | to add |
| 07 | nnU-Net hybrid (FLAIR, T1, map), fold 0 | `nnunet_hybride/` | to add |
| 08 | Held-out evaluation | `test_final/` | to add |
| 09 | Map quality (step 1) and 5-fold cross-validation | `cv_out/` | `09_cv_5folds_and_map_quality.ipynb` |

The execution logs of every run are in `../logs/`.
