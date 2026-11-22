"""English versions of the two diagnostic figures of the diffusion model, built from the Kaggle figures.

Sources (figures/sources/): the figures produced by the notebook "FCD - Diagnostic du modèle de diffusion".
Currently screen captures of the Kaggle viewer (low resolution). If the original PNG files are downloaded,
put them in figures/sources/ as fig_exemple_synthetique.png and fig_profils.png and run with --originals
(the crop boxes are then detected automatically, see below).
Only text is replaced (French labels -> English); image content and data points are unchanged.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC, F = ROOT / 'figures' / 'sources', ROOT / 'figures'
plt.rcParams.update({'font.size': 8.5, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})


def crop(im, box):
    return np.asarray(im.crop(box))


# ---------------------------------------------------------------- synthetic-lesion example (4 panels)
im = Image.open(SRC / 'capture_fig_exemple_synthetique.png').convert('RGB')
boxes = [(7, 95, 161, 291), (203, 95, 358, 291), (400, 95, 555, 291), (597, 95, 751, 291)]
titles = ['FLAIR with inserted lesion\n(control sub-00002)', 'Pseudo-healthy\nreconstruction (DDPM)',
          '|FLAIR $-$ reconstruction|', 'Initial anomaly map\n(smoothed, FLAIR + T1)']
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.35))
for a, b, t in zip(ax, boxes, titles):
    a.imshow(crop(im, b)); a.set_title(t, fontsize=8); a.axis('off')
fig.tight_layout(w_pad=0.6)
fig.savefig(F / 'fig_ddpm_synthetic_example.pdf', bbox_inches='tight', dpi=300)
fig.savefig(F / 'fig_ddpm_synthetic_example.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- score profiles in healthy controls
im = Image.open(SRC / 'capture_fig_profils.png').convert('RGB')
left = np.array(im.crop((20, 85, 385, 294)))
right = np.array(im.crop((385, 85, 758, 294))).copy()
right[5:40, 33:166] = 255                          # remove the French legend (redrawn in English below)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for a, img in zip(ax, (left, right)):
    a.imshow(img); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
ax[0].set_xlabel('Distance from the brain boundary (mm)')
ax[0].set_ylabel('Mean score, healthy controls')
ax[0].set_title('(a) Score vs distance to the brain boundary', loc='left', fontsize=8.5)
ax[1].set_xlabel('Decile of FLAIR gradient magnitude')
ax[1].set_title('(b) Score vs local contrast', loc='left', fontsize=8.5)
ax[1].scatter([], [], c='red', s=12, label='FCD lesions (dev. A)')
ax[1].errorbar([], [], yerr=[], fmt='o-', color='#1f77b4', ms=4, label='healthy (mean ± SD)')
ax[1].legend(loc='upper left', bbox_to_anchor=(0.085, 0.99), fontsize=6, frameon=False, handlelength=1.2, borderpad=0.2, labelspacing=0.25)
fig.tight_layout(w_pad=0.8)
fig.savefig(F / 'fig_ddpm_profiles.pdf', bbox_inches='tight', dpi=300)
fig.savefig(F / 'fig_ddpm_profiles.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print('OK: fig_ddpm_synthetic_example, fig_ddpm_profiles')

# ---------------------------------------------------------------- training curves (values printed in the verifica log)
import pandas as pd
D = ROOT / 'data'
dd = pd.read_csv(D / 'ddpm_loss_depuis_journal.csv'); ae = pd.read_csv(D / 'ae_loss_depuis_journal.csv')
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e5e1', 'grid.linewidth': 0.6})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax[0].plot(dd.step / 1000, dd.loss, color='#2a78d6', lw=2, marker='o', ms=3)
ax[0].set_yscale('log'); ax[0].set_xlabel('Training step (thousands)'); ax[0].set_ylabel('Noise-prediction MSE (log)')
ax[0].set_title(f'(a) DDPM: {20569:,} steps in 3.0 h (T4)', loc='left', fontsize=8.5)
ax[1].plot(ae.step / 1000, ae.loss, color='#9a9994', lw=2, marker='o', ms=3)
ax[1].set_xlabel('Training step (thousands)'); ax[1].set_ylabel('Reconstruction MSE')
ax[1].set_title(f'(b) Reference autoencoder: {56128:,} steps in 30 min', loc='left', fontsize=8.5)
fig.tight_layout(w_pad=1.2)
fig.savefig(F / 'fig_ddpm_training.pdf', bbox_inches='tight'); fig.savefig(F / 'fig_ddpm_training.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('OK: fig_ddpm_training')

# ---------------------------------------------------------------- initial vs retained score (one development patient)
im = Image.open(SRC / 'capture_fig_variantes.png').convert('RGB')
boxes = [(4, 20, 214, 270), (254, 20, 476, 270), (514, 20, 736, 270)]
titles = ['FLAIR (sub-00098, development A)', 'Initial score\n(|residual| FLAIR + T1, 2 draws)',
          'Retained score\n(positive FLAIR residual, 6 draws,\ncontrast-normalised)']
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.75))
for a, b, t in zip(ax, boxes, titles):
    a.imshow(crop(im, b)); a.set_title(t, fontsize=7.5); a.axis('off')
fig.tight_layout(w_pad=0.5)
for ext in ('pdf', 'png'):
    fig.savefig(F / f'fig_score_variants.{ext}', bbox_inches='tight', dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- held-out example with the largest gain
im = Image.open(SRC / 'capture_fig_qualitative_test.png').convert('RGB')
boxes = [(22, 45, 306, 400), (400, 45, 748, 400)]
titles = ['FLAIR: label (green), baseline (blue), hybrid (red)\nDice baseline 0.01, hybrid 0.44',
          'Anomaly map (z-score) with label (green)']
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={'width_ratios': [284, 348]})
for a, b, t in zip(ax, boxes, titles):
    a.imshow(crop(im, b)); a.set_title(t, fontsize=7.5); a.axis('off')
fig.tight_layout(w_pad=0.5)
for ext in ('pdf', 'png'):
    fig.savefig(F / f'fig_qualitative.{ext}', bbox_inches='tight', dpi=300)
plt.close(fig)
print('OK: fig_score_variants, fig_qualitative')

# ---------------------------------------------------------------- preprocessing quality control (two flagged patients)
im = Image.open(SRC / 'capture_fig_qc_preproc.png').convert('RGB')
rows = [(12, 216), (232, 440)]
cols = [(12, 198), (226, 412), (436, 622)]
ttl = [['sub-00018: FLAIR, brain mask (yellow), label (red)', 'Registered T1 + FLAIR edges (cyan)', 'Residual T1 bias after N4'],
       ['sub-00015: FLAIR, brain mask (yellow), label (red)', 'Registered T1 + FLAIR edges (cyan)', 'Residual T1 bias after N4']]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.0))
for i, (y0, y1) in enumerate(rows):
    for j, (x0, x1) in enumerate(cols):
        ax[i, j].imshow(crop(im, (x0, y0, x1, y1))); ax[i, j].set_title(ttl[i][j], fontsize=6.5); ax[i, j].axis('off')
fig.tight_layout(h_pad=0.6, w_pad=0.4)
for ext in ('pdf', 'png'):
    fig.savefig(F / f'fig_qc_preproc.{ext}', bbox_inches='tight', dpi=300)
plt.close(fig)
print('OK: fig_qc_preproc')
