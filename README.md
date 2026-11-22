# fcd-epileptic-diffusion-nnunet

**Diffusion-model anomaly maps as input priors for nnU-Net segmentation of focal cortical dysplasia (FCD) type II**, on the public OpenNeuro dataset [ds004199](https://openneuro.org/datasets/ds004199).

## Pipeline

1. **Preprocessing** of the 170 subjects: FLAIR reference grid at 1 mm, rigid T1→FLAIR registration, HD-BET brain extraction, N4 bias-field correction, intensity normalisation, cropping, quantitative and visual quality control.
2. **Generative model**: a 2D denoising diffusion probabilistic model (DDPM; U-Net, 12.2 M parameters, T = 1000) trained on FLAIR and T1 slices of 63 healthy controls; a convolutional autoencoder as reference.
3. **Anomaly map**: partial noising to t* = 150, 25 DDIM steps, mean of 6 draws; positive FLAIR residual, smoothed and z-scored against healthy controls per decile of local FLAIR contrast. The score was diagnosed on synthetic lesions and selected on development patients only.
4. **Segmentation**: nnU-Net v2.8.1 `3d_fullres`, baseline (FLAIR, T1) vs hybrid (FLAIR, T1, anomaly map), identical plans and training budget.
5. **Evaluation**: pre-specified criteria, patient-level 5-fold cross-validation (50 patients), held-out set evaluated once (27 patients), map ablation (map set to zero), map-quality analysis.

## Main results

| Analysis | n | Dice baseline | Dice hybrid | Difference [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|
| 5-fold cross-validation, out-of-fold | 50 | 0.434 | 0.372 | −0.063 [−0.115, −0.013] | 0.0027 |
| Held-out set | 27 | 0.262 | 0.199 | −0.063 [−0.157, 0.022] | 0.408 |

The hybrid model relies on the map (Dice 0.372 → 0.220 when the map is set to zero), but the map alone is weakly informative and its quality does not predict the gain (Spearman ρ = 0.00).

## Repository structure

| Path | Content |
|---|---|
| `notebooks/` | Kaggle notebooks, in execution order (see `notebooks/README.md`) |
| `data/` | per-patient result files (cross-validation, held-out set, map quality, ablation) and values extracted from the logs |
| `scripts/analyse_et_figures.py` | recomputes all statistics, tables and result figures from `data/` |
| `scripts/figures_diagnostic_diffusion.py` | figures of the diffusion model and of preprocessing quality control |
| `figures/sources/` | notebook figures used as inputs by the figure script |
| `logs/` | execution logs of every notebook run and notebook figures (`logs/INDEX.md`) |

## Reproduce the statistics

```bash
pip install -r requirements.txt
python scripts/analyse_et_figures.py
python scripts/figures_diagnostic_diffusion.py
```

Outputs are written to `generated/` (values and tables) and `figures/`. The script checks the integrity of the inputs (folds, roles, stored differences) and uses the same bootstrap seed as the notebooks, so confidence intervals match the logs exactly.

## Data

The MRI data are not redistributed. They are available on OpenNeuro (ds004199; Schuch et al., 2023, *Scientific Data*); the preprocessing notebook downloads them directly from the OpenNeuro server.

## Licence

MIT (see `LICENSE`). Citation information is in `CITATION.cff`.
