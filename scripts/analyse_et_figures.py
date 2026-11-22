"""Recompute every statistic, table and result figure of the study from the per-patient result files in ../data.

Usage (from the project root):  python scripts/analyse_et_figures.py
Outputs: generated/numbers.tex (LaTeX macros used in main.tex), generated/tab_*.tex, figures/*.pdf,
         generated/verification.json (all values, unrounded).
No number in main.tex is typed by hand: every result is a macro defined in generated/numbers.tex.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D, G, F = ROOT / 'data', ROOT / 'generated', ROOT / 'figures'
G.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
N_BOOT = 10000
COL = {'A': '#2a78d6', 'B': '#eb6834'}      # validated categorical pair (protocol A, protocol B)
MRK = {'A': 'o', 'B': '^'}                  # secondary encoding (print / colour-vision deficiency)
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
                     'legend.frameon': False, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})


# ----------------------------------------------------------------------------- statistics (same as notebooks)
def boot_ci(d, n=N_BOOT, seed=0):
    v = np.asarray(d, float); rs = np.random.RandomState(seed)
    m = v[rs.randint(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def hodges_lehmann(d):
    v = np.asarray(d, float); w = (v[:, None] + v[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(v))]))

def rank_biserial(d):
    v = np.asarray(d, float); v = v[v != 0]
    r = stats.rankdata(np.abs(v))
    return float((r[v > 0].sum() - r[v < 0].sum()) / r.sum()) if len(v) else 0.0

def paired(x, y, lower_is_better=False):
    d = pd.Series(np.asarray(x, float) - np.asarray(y, float))
    p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    lo, hi = boot_ci(d)
    better = (d < 0) if lower_is_better else (d > 0)
    return dict(mean=float(d.mean()), median=float(d.median()), hl=hodges_lehmann(d), lo=lo, hi=hi, p=p,
                rrb=rank_biserial(d), n_better=int(better.sum()), n_change=int((d != 0).sum()), n=len(d))

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw

def mcnemar(b_det, h_det):
    gain = int((~b_det & h_det).sum()); lost = int((b_det & ~h_det).sum())
    p = float(stats.binomtest(gain, gain + lost).pvalue) if gain + lost else 1.0
    return gain, lost, p

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); cur = 0.0
    for i, k in enumerate(o):
        cur = max(cur, (m - i) * ps[k]); adj[k] = min(1.0, cur)
    return adj

def spearman_ci(x, y, n=5000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float); rho, p = stats.spearmanr(x, y)
    rs = np.random.RandomState(seed); bs = []
    for _ in range(n):
        i = rs.randint(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0: bs.append(stats.spearmanr(x[i], y[i])[0])
    return dict(rho=float(rho), p=float(p), lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)))


# ----------------------------------------------------------------------------- load and check the source files
oof = pd.read_csv(D / 'cv_oof_par_patient.csv', header=[0, 1], index_col=0)
B, H = oof['baseline'].copy(), oof['hybride'].copy()
for X in (B, H):
    for c in ('detecte', 'hit1', 'irm_negatif'):
        X[c] = X[c].astype(str).str.strip().map({'True': True, 'False': False})
Z = pd.read_csv(D / 'cv_hybride_sans_carte_oof.csv', index_col=0).loc[B.index]
Z['detecte'] = Z['detecte'].astype(bool)
Q = pd.read_csv(D / 'qualite_carte.csv', index_col=0)
T = pd.read_csv(D / 'etape1_gain_vs_qualite_carte.csv', index_col=0)
T = T[T.jeu == 'test'].copy()
splits = json.load(open(D / 'splits_cv.json'))
crit_cv = json.load(open(D / 'criteres_cv.json'))
crit_test = json.load(open(D / 'criteres_test_depuis_journal.json'))
sel = pd.read_csv(D / 'selection_variante_carte_depuis_journal.csv')

assert len(B) == 50 and len(T) == 27 and not set(B.index) & set(T.index)
assert (B.fold == H.fold).all() and (Z.fold == B.fold).all()
for k, s in enumerate(splits):
    assert set(B.index[B.fold == k]) == set(s['val']) and not set(s['train']) & set(s['val'])
assert sorted(sum((s['val'] for s in splits), [])) == sorted(B.index)
delta_file = pd.read_csv(D / 'cv_delta_vs_qualite_carte.csv', index_col=0)
assert np.allclose(delta_file.loc[B.index, 'delta'], H.dice - B.dice, atol=1e-12)

# ----------------------------------------------------------------------------- analyses
R = {}
R['cv_main'] = paired(H.dice, B.dice)
R['cv_auprc'] = paired(H.auprc, B.auprc)
R['cv_fp'] = paired(H.n_fp_clusters, B.n_fp_clusters, lower_is_better=True)
g, l, p = mcnemar(B.detecte.values, H.detecte.values)
R['cv_det'] = dict(b=int(B.detecte.sum()), h=int(H.detecte.sum()), gain=g, lost=l, p=p,
                   b_ci=wilson(int(B.detecte.sum()), 50), h_ci=wilson(int(H.detecte.sum()), 50))
adj = holm([R['cv_auprc']['p'], R['cv_fp']['p'], R['cv_det']['p']])
R['cv_auprc']['holm'], R['cv_fp']['holm'], R['cv_det']['holm'] = map(float, adj)
R['cv_abl'] = paired(H.dice, Z.dice)
R['cv_abl_det'] = int(Z.detecte.sum())
R['cv_h2'] = spearman_ci(Q.loc[B.index, 'map_auprc'], H.dice - B.dice)
for k in ('map_auroc', 'map_lift', 'map_cohen_d'):
    rho, pp = stats.spearmanr(Q.loc[B.index, k], H.dice - B.dice); R[f'cv_sens_{k}'] = dict(rho=float(rho), p=float(pp))
folds = pd.DataFrame({'n': B.groupby('fold').size(), 'b': B.groupby('fold').dice.mean(),
                      'h': H.groupby('fold').dice.mean(), 'z': Z.groupby('fold').dice.mean()})
folds['d'] = folds.h - folds.b
R['folds'] = folds.reset_index().to_dict('records')
for pr in 'AB':
    s = B.protocole == pr
    R[f'cv_prot_{pr}'] = {**paired(H.dice[s], B.dice[s]), 'b_mean': float(B.dice[s].mean()), 'h_mean': float(H.dice[s].mean())}
R['cv_means'] = dict(b=float(B.dice.mean()), b_sd=float(B.dice.std()), h=float(H.dice.mean()), h_sd=float(H.dice.std()),
                     z=float(Z.dice.mean()), b_med=float(B.dice.median()), h_med=float(H.dice.median()))
R['cv_mri_neg'] = list(B.index[B.irm_negatif])

R['test_main'] = paired(T.dice_hybride, T.dice_baseline)
R['test_abl'] = paired(T.dice_hybride, T.dice_sans_carte)
g, l, p = mcnemar(T.det_b.values.astype(bool), T.det_h.values.astype(bool))
R['test_det'] = dict(b=int(T.det_b.sum()), h=int(T.det_h.sum()), gain=g, lost=l, p=p)
R['test_means'] = dict(b=float(T.dice_baseline.mean()), b_sd=float(T.dice_baseline.std()), h=float(T.dice_hybride.mean()),
                       h_sd=float(T.dice_hybride.std()), z=float(T.dice_sans_carte.mean()))
for pr in 'AB':
    s = T.protocole == pr
    R[f'test_prot_{pr}'] = {**paired(T.dice_hybride[s], T.dice_baseline[s]), 'n': int(s.sum()),
                            'b_mean': float(T.dice_baseline[s].mean()), 'h_mean': float(T.dice_hybride[s].mean())}
R['test_h2'] = spearman_ci(T.map_auprc, T.delta)

qg = Q.groupby(['split', 'protocole'])[['map_auprc', 'map_auroc', 'map_cohen_d', 'map_lift', 'map_top1', 'map_top3', 'map_n_fp']].mean()
R['q_groups'] = {f'{a}|{b}': v for (a, b), v in qg.to_dict('index').items()}
R['q_mw_split'] = float(stats.mannwhitneyu(Q.loc[Q.split == 'entrainement', 'map_auprc'], Q.loc[Q.split == 'test', 'map_auprc']).pvalue)
R['q_mw_prot'] = float(stats.mannwhitneyu(Q.loc[Q.protocole == 'A', 'map_auprc'], Q.loc[Q.protocole == 'B', 'map_auprc']).pvalue)
for pr in 'AB':
    R[f'q_mw_valtest_{pr}'] = float(stats.mannwhitneyu(Q.loc[(Q.protocole == pr) & (Q.split == 'entrainement'), 'map_auprc'],
                                                       Q.loc[(Q.protocole == pr) & (Q.split == 'test'), 'map_auprc']).pvalue)
R['q_all_auprc_mean'] = float(Q.map_auprc.mean()); R['q_all_auroc_mean'] = float(Q.map_auroc.mean())
R['q_prev_mean'] = float(Q.prevalence.mean())
R['q_top1_n'] = int(Q.map_top1.sum()); R['q_top3_n'] = int(Q.map_top3.sum()); R['q_n'] = len(Q)
R['q_nfp_mean'] = float(Q.map_n_fp.mean())
R['crit_cv_time'] = crit_cv['horodatage']; R['crit_test_time'] = crit_test['horodatage']
json.dump(R, open(G / 'verification.json', 'w'), indent=1, default=float)


# ----------------------------------------------------------------------------- LaTeX macros
def f3(x): return f'{x:.3f}'
def s3(x): return '0.000' if abs(x) < 5e-4 else f'{x:+.3f}'.replace('-', '\\ensuremath{-}')
def f2(x): return f'{x:.2f}'
def s2(x): return f'{x:+.2f}'.replace('-', '\\ensuremath{-}')
def pv(p): return '<0.001' if p < 0.001 else (f'{p:.4f}' if p < 0.01 else f'{p:.3f}')

M = {}
def put(name, val): M[name] = val
cm, cv = R['cv_means'], R['cv_main']
put('CVn', '50'); put('CVbase', f3(cm['b'])); put('CVbaseSD', f3(cm['b_sd'])); put('CVhyb', f3(cm['h'])); put('CVhybSD', f3(cm['h_sd']))
put('CVdiff', s3(cv['mean'])); put('CVlo', s3(cv['lo'])); put('CVhi', s3(cv['hi'])); put('CVp', pv(cv['p']))
put('CVhl', s3(cv['hl'])); put('CVmed', s3(cv['median'])); put('CVrrb', s2(cv['rrb'])); put('CVbetter', str(cv['n_better'])); put('CVchange', str(cv['n_change']))
put('CVfoldsBetter', str(int((folds.d > 0).sum()))); put('CVfoldSD', f3(folds.d.std()))
put('CVauprcDiff', s3(R['cv_auprc']['mean'])); put('CVauprcLo', s3(R['cv_auprc']['lo'])); put('CVauprcHi', s3(R['cv_auprc']['hi']))
put('CVauprcP', pv(R['cv_auprc']['p'])); put('CVauprcHolm', pv(R['cv_auprc']['holm']))
put('CVfpDiff', s2(R['cv_fp']['mean'])); put('CVfpLo', s2(R['cv_fp']['lo'])); put('CVfpHi', s2(R['cv_fp']['hi']))
put('CVfpP', pv(R['cv_fp']['p'])); put('CVfpHolm', pv(R['cv_fp']['holm']))
dt = R['cv_det']
put('CVdetB', str(dt['b'])); put('CVdetH', str(dt['h'])); put('CVdetGain', str(dt['gain'])); put('CVdetLost', str(dt['lost']))
put('CVdetP', pv(dt['p'])); put('CVdetHolm', pv(dt['holm']))
put('CVdetBlo', f'{100*dt["b_ci"][0]:.0f}'); put('CVdetBhi', f'{100*dt["b_ci"][1]:.0f}')
put('CVdetHlo', f'{100*dt["h_ci"][0]:.0f}'); put('CVdetHhi', f'{100*dt["h_ci"][1]:.0f}')
ab = R['cv_abl']
put('CVzero', f3(cm['z'])); put('CVablDiff', s3(ab['mean'])); put('CVablLo', s3(ab['lo'])); put('CVablHi', s3(ab['hi'])); put('CVablP', pv(ab['p']))
put('CVzeroDet', str(R['cv_abl_det']))
h2 = R['cv_h2']; put('HtwoRho', s2(h2['rho']) if abs(h2['rho']) >= 0.005 else '0.00'); put('HtwoLo', s2(h2['lo'])); put('HtwoHi', s2(h2['hi'])); put('HtwoP', pv(h2['p']))
for k, nm in (('map_auroc', 'Auroc'), ('map_lift', 'Lift'), ('map_cohen_d', 'Cohen')):
    put(f'Sens{nm}Rho', s2(R[f'cv_sens_{k}']['rho'])); put(f'Sens{nm}P', pv(R[f'cv_sens_{k}']['p']))
for pr in 'AB':
    r = R[f'cv_prot_{pr}']
    put(f'CVprot{pr}b', f3(r['b_mean'])); put(f'CVprot{pr}h', f3(r['h_mean'])); put(f'CVprot{pr}d', s3(r['mean']))
    put(f'CVprot{pr}lo', s3(r['lo'])); put(f'CVprot{pr}hi', s3(r['hi'])); put(f'CVprot{pr}p', pv(r['p']))
tm, tt = R['test_means'], R['test_main']
put('Tbase', f3(tm['b'])); put('TbaseSD', f3(tm['b_sd'])); put('Thyb', f3(tm['h'])); put('ThybSD', f3(tm['h_sd']))
put('Tdiff', s3(tt['mean'])); put('Tlo', s3(tt['lo'])); put('Thi', s3(tt['hi'])); put('Tp', pv(tt['p'])); put('Thl', s3(tt['hl']))
put('Trrb', s2(tt['rrb'])); put('Tbetter', str(tt['n_better'])); put('Tchange', str(tt['n_change']))
td = R['test_det']; put('TdetB', str(td['b'])); put('TdetH', str(td['h'])); put('TdetGain', str(td['gain'])); put('TdetLost', str(td['lost'])); put('TdetP', pv(td['p']))
ta = R['test_abl']; put('Tzero', f3(tm['z'])); put('TablDiff', s3(ta['mean'])); put('TablLo', s3(ta['lo'])); put('TablHi', s3(ta['hi'])); put('TablP', pv(ta['p']))
for pr in 'AB':
    r = R[f'test_prot_{pr}']
    put(f'Tprot{pr}n', str(r['n'])); put(f'Tprot{pr}b', f3(r['b_mean'])); put(f'Tprot{pr}h', f3(r['h_mean'])); put(f'Tprot{pr}d', s3(r['mean'])); put(f'Tprot{pr}p', pv(r['p']))
th2 = R['test_h2']; put('THtwoRho', s2(th2['rho'])); put('THtwoP', pv(th2['p']))
put('QauprcAll', f3(R['q_all_auprc_mean'])); put('QaurocAll', f2(R['q_all_auroc_mean'])); put('Qprev', f'{R["q_prev_mean"]:.4f}')
put('QtopOne', str(R['q_top1_n'])); put('QtopThree', str(R['q_top3_n'])); put('Qn', str(R['q_n'])); put('Qnfp', f'{R["q_nfp_mean"]:.0f}')
for key, nm in (('entrainement|A', 'TrA'), ('entrainement|B', 'TrB'), ('test|A', 'TeA'), ('test|B', 'TeB')):
    put(f'Q{nm}', f3(R['q_groups'][key]['map_auprc']))
put('QmwSplit', pv(R['q_mw_split'])); put('QmwProt', pv(R['q_mw_prot'])); put('QmwA', pv(R['q_mw_valtest_A'])); put('QmwB', pv(R['q_mw_valtest_B']))
put('CritTestTime', crit_test['horodatage']); put('CritCVTime', crit_cv['horodatage'])
put('SelAuprcInit', f'{sel.set_index("variant").loc["abs_FT_2", "auprc_mean_fcdA_val"]:.4f}')
put('SelAuprcFinal', f'{sel.set_index("variant").loc["posF_6_norm", "auprc_mean_fcdA_val"]:.4f}')


# ----------------------------------------------------------------------------- additions v3 (test secondary, step 1, selection, controls)
import statsmodels.formula.api as smf
TB = pd.read_csv(D / 'test_baseline_dice_auprc.csv', index_col=0).loc[T.index]
TH = pd.read_csv(D / 'test_hybride_dice_auprc.csv', index_col=0).loc[T.index]
assert np.allclose(TB.dice, T.dice_baseline) and np.allclose(TH.dice, T.dice_hybride)
ta2 = paired(TH.auprc, TB.auprc)
put('TauprcB', f3(TB.auprc.mean())); put('TauprcH', f3(TH.auprc.mean())); put('TauprcDiff', s3(ta2['mean']))
put('TauprcLo', s3(ta2['lo'])); put('TauprcHi', f'{ta2["hi"]:+.4f}'.replace('-', '\\ensuremath{-}')); put('TauprcP', pv(ta2['p']))
TS = json.load(open(D / 'test_secondaires_depuis_journal.json'))
put('TfpDiff', s2(TS['fp_diff'])); put('TfpLo', s2(TS['fp_lo'])); put('TfpHi', s2(TS['fp_hi'])); put('TfpP', pv(TS['fp_p']))
put('ValZeroDiff', s3(TS['validation_fold0_before_test_criteria']['diff'])); put('ValZeroP', pv(TS['validation_fold0_before_test_criteria']['p']))
E = pd.read_csv(D / 'etape1_gain_vs_qualite_carte.csv')
assert len(E) == 37
for nm, col in (('Gain', 'delta'), ('Rel', 'dependance')):
    rho, pp = stats.spearmanr(E.map_auprc, E[col]); put(f'Step{nm}Rho', s2(rho)); put(f'Step{nm}P', pv(pp))
E['map_auprc_z'] = (E.map_auprc - E.map_auprc.mean()) / E.map_auprc.std()
reg = smf.ols('delta ~ map_auprc_z + dice_baseline + C(protocole) + C(jeu)', E).fit(cov_type='HC3')
for k, nm in (('dice_baseline', 'Base'), ('C(protocole)[T.B]', 'Prot'), ('map_auprc_z', 'Map')):
    put(f'Reg{nm}', s3(reg.params[k])); put(f'Reg{nm}P', pv(reg.pvalues[k]))
put('RegN', str(int(reg.nobs)))
C7 = pd.read_csv(D / 'controles_A.csv')
cc = C7.groupby('variante').n_clusters.mean()
put('CtrlClustInit', f'{cc["abs_FT_2"]:.2f}'); put('CtrlClustFinal', f'{cc["posF_6_norm"]:.2f}'); put('CtrlN', str(C7.participant_id.nunique()))
roles = pd.read_csv(D / 'roles_sujets_depuis_journal.csv').set_index('role').n
put('HCtrain', str(roles['hc_train'])); put('HCcal', str(roles['hc_calibration'])); put('HCcheck', str(roles['controle_t1_sag111']))
put('Nexcl', str(roles['exclu_autre_protocole']))
assert roles.sum() == 170

with open(G / 'numbers.tex', 'w') as fh:
    fh.write('% AUTO-GENERATED by scripts/analyse_et_figures.py from data/*. Do not edit by hand.\n')
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')


# ----------------------------------------------------------------------------- tables
def ci(r, fmt=s3): return f'{fmt(r["mean"])} [{fmt(r["lo"])}, {fmt(r["hi"])}]'
rows = [
    ('Dice, mean $\\pm$ SD', f'{f3(cm["b"])} $\\pm$ {f3(cm["b_sd"])}', f'{f3(cm["h"])} $\\pm$ {f3(cm["h_sd"])}', ci(cv), pv(cv['p']), '--'),
    ('Voxel-wise AUPRC', f3(B.auprc.mean()), f3(H.auprc.mean()), ci(R['cv_auprc']), pv(R['cv_auprc']['p']), pv(R['cv_auprc']['holm'])),
    ('False-positive clusters / patient', f2(B.n_fp_clusters.mean()), f2(H.n_fp_clusters.mean()), ci(R['cv_fp'], s2), pv(R['cv_fp']['p']), pv(R['cv_fp']['holm'])),
    ('Detection, $k$/50', f'{dt["b"]}/50', f'{dt["h"]}/50', f'+{dt["gain"]} / \\ensuremath{{-}}{dt["lost"]}', pv(dt['p']), pv(dt['holm'])),
]
with open(G / 'tab_cv.tex', 'w') as fh:
    fh.write('\\begin{tabular}{lccccc}\n\\toprule\nEndpoint & Baseline & Hybrid & Difference [95\\% CI] & $p$ & $p_{\\mathrm{Holm}}$\\\\\n\\midrule\n')
    for r in rows: fh.write(' & '.join(r) + '\\\\\n')
    fh.write('\\botrule\n\\end{tabular}\n')

with open(G / 'tab_folds.tex', 'w') as fh:
    fh.write('\\begin{tabular}{cccccc}\n\\toprule\nFold & $n$ & Baseline & Hybrid & Hybrid $-$ baseline & Hybrid, map set to zero\\\\\n\\midrule\n')
    for k, r in folds.iterrows():
        fh.write(f'{k} & {int(r.n)} & {f3(r.b)} & {f3(r.h)} & {s3(r.d)} & {f3(r.z)}\\\\\n')
    fh.write(f'\\midrule\nAll (out-of-fold) & 50 & {f3(cm["b"])} & {f3(cm["h"])} & {s3(cv["mean"])} & {f3(cm["z"])}\\\\\n')
    fh.write('\\botrule\n\\end{tabular}\n')

with open(G / 'tab_cv_test.tex', 'w') as fh:
    fh.write('\\begin{tabular}{lcccccc}\n\\toprule\nAnalysis & $n$ & Baseline & Hybrid & Difference [95\\% CI] & $p$ & Detection (B vs H)\\\\\n\\midrule\n')
    fh.write(f'5-fold CV, out-of-fold & 50 & {f3(cm["b"])} & {f3(cm["h"])} & {ci(cv)} & {pv(cv["p"])} & {dt["b"]}/50 vs {dt["h"]}/50\\\\\n')
    fh.write(f'Held-out test set & 27 & {f3(tm["b"])} & {f3(tm["h"])} & {ci(tt)} & {pv(tt["p"])} & {td["b"]}/27 vs {td["h"]}/27\\\\\n')
    fh.write('\\midrule\n')
    for pr in 'AB':
        r = R[f'cv_prot_{pr}']
        fh.write(f'\\quad CV, protocol {pr} & 25 & {f3(r["b_mean"])} & {f3(r["h_mean"])} & {ci(r)} & {pv(r["p"])} & --\\\\\n')
    for pr in 'AB':
        r = R[f'test_prot_{pr}']
        fh.write(f'\\quad Test, protocol {pr} & {r["n"]} & {f3(r["b_mean"])} & {f3(r["h_mean"])} & {ci(r)} & {pv(r["p"])} & --\\\\\n')
    fh.write('\\botrule\n\\end{tabular}\n')

with open(G / 'tab_ablation.tex', 'w') as fh:
    fh.write('\\begin{tabular}{lccccc}\n\\toprule\nAnalysis & $n$ & Hybrid & Hybrid, map $=0$ & Difference [95\\% CI] & $p$\\\\\n\\midrule\n')
    fh.write(f'5-fold CV & 50 & {f3(cm["h"])} & {f3(cm["z"])} & {ci(ab)} & {pv(ab["p"])}\\\\\n')
    fh.write(f'Held-out test & 27 & {f3(tm["h"])} & {f3(tm["z"])} & {ci(ta)} & {pv(ta["p"])}\\\\\n')
    fh.write('\\botrule\n\\end{tabular}\n')

with open(G / 'tab_map_quality.tex', 'w') as fh:
    fh.write('\\begin{tabular}{llcccccc}\n\\toprule\nSet & Protocol & $n$ & AUPRC & AUROC & Cohen\'s $d$ & Top-1 hit & Top-3 hit\\\\\n\\midrule\n')
    for (sp, pr), g_ in Q.groupby(['split', 'protocole']):
        name = 'Development (CV)' if sp == 'entrainement' else 'Held-out test'
        fh.write(f'{name} & {pr} & {len(g_)} & {f3(g_.map_auprc.mean())} & {f3(g_.map_auroc.mean())} & {f3(g_.map_cohen_d.mean())} & '
                 f'{int(g_.map_top1.sum())}/{len(g_)} & {int(g_.map_top3.sum())}/{len(g_)}\\\\\n')
    fh.write('\\botrule\n\\end{tabular}\n')

lab = {'abs_FT_2': '$|r_F|,|r_{T1}|$, 2 draws (initial)', 'abs_FT_6': '$|r_F|,|r_{T1}|$, 6 draws', 'posF_2': '$r_F^{+}$, 2 draws',
       'posF_6': '$r_F^{+}$, 6 draws', 'posF_6_norm': '$r_F^{+}$, 6 draws, normalised (retained)'}
with open(G / 'tab_selection.tex', 'w') as fh:
    fh.write('\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{2}{c}{\\shortstack{Synthetic lesions\\\\AUPRC}} & \\multicolumn{2}{c}{\\shortstack{FCD, dev.~A\\\\($n=25$)}} & Healthy\\\\\n'
             '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nScore & $+1.5$ SD & $+3$ SD & mean AUPRC & top-3 hit & clusters\\\\\n\\midrule\n')
    for _, r in sel.iterrows():
        fh.write(f'{lab[r.variant]} & {r.synth_auprc_1p5sd:.3f} & {r.synth_auprc_3sd:.3f} & {r.auprc_mean_fcdA_val:.4f} & {r.top3_hit:.2f} & {r.clusters_healthy:.1f}\\\\\n')
    fh.write('\\botrule\n\\end{tabular}\n')


# ----------------------------------------------------------------------------- diagnosis of the diffusion model
syn = pd.read_csv(D / 'test_lesions_synthetiques_diagnostic.csv')
assert len(syn) == 36 and syn.sujet.nunique() == 6
gs = syn.groupby(['amplitude_z', 'rayon_vox', 'methode'])[['auprc', 'auroc', 'hit1', 'hit3']].mean()
with open(G / 'tab_ddpm_ae.tex', 'w') as fh:
    fh.write('\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{3}{c}{DDPM} & \\multicolumn{3}{c}{Autoencoder}\\\\\n'
             '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nSynthetic lesion & AUPRC & AUROC & top-3 hit & AUPRC & AUROC & top-3 hit\\\\\n\\midrule\n')
    for amp, rad in [(3.0, 6), (1.5, 6), (1.5, 3)]:
        d_, a_ = gs.loc[(amp, rad, 'ddpm')], gs.loc[(amp, rad, 'ae')]
        fh.write(f'$+{amp:g}$ SD, radius {rad} mm & {d_.auprc:.3f} & {d_.auroc:.3f} & {int(round(6*d_.hit3))}/6 & '
                 f'{a_.auprc:.3f} & {a_.auroc:.3f} & {int(round(6*a_.hit3))}/6\\\\\n')
    fh.write('\\botrule\n\\end{tabular}\n')

ts = pd.read_csv(D / 'selection_tstar_depuis_journal.csv')
nm = {'ddpm_t150': 'DDPM, $t^\\ast=150$ (retained)', 'ddpm_t250': 'DDPM, $t^\\ast=250$', 'ddpm_t350': 'DDPM, $t^\\ast=350$', 'ae': 'Autoencoder'}
with open(G / 'tab_tstar.tex', 'w') as fh:
    fh.write('\\begin{tabular}{lcccccc}\n\\toprule\nModel & \\shortstack{mean\\\\AUPRC} & \\shortstack{median\\\\AUPRC} & \\shortstack{mean\\\\AUROC} & \\shortstack{top-3\\\\hit} & \\shortstack{lesion\\\\percentile} & \\shortstack{clusters,\\\\healthy}\\\\\n\\midrule\n')
    for _, r in ts.iterrows():
        fh.write(f'{nm[r.methode]} & {r.auprc_moy:.3f} & {r.auprc_med:.3f} & {r.auroc_moy:.3f} & {r.hit3:.2f} & {r.lesion_pctl_med:.3f} & {r.clusters_sains:.1f}\\\\\n')
    fh.write('\\botrule\n\\end{tabular}\n')
rsc = pd.read_csv(D / 'rescoring_gradient_depuis_csv_et_journal.csv')
with open(G / 'numbers.tex', 'a') as fh:
    for amp, rad, nm in [(3.0, 6, 'Big'), (1.5, 6, 'Mid'), (1.5, 3, 'Small')]:
        fh.write(f'\\newcommand{{\\Syn{nm}Ddpm}}{{{gs.loc[(amp, rad, "ddpm")].auprc:.3f}}}\n')
        fh.write(f'\\newcommand{{\\Syn{nm}Ae}}{{{gs.loc[(amp, rad, "ae")].auprc:.3f}}}\n')
    fh.write(f'\\newcommand{{\\RscMin}}{{{rsc.auprc_moy.min():.3f}}}\n\\newcommand{{\\RscMax}}{{{rsc.auprc_moy.max():.3f}}}\n')
    fh.write(f'\\newcommand{{\\RscClustA}}{{{rsc.clusters_sains.iloc[0]:.1f}}}\n\\newcommand{{\\RscClustB}}{{{rsc.clusters_sains.iloc[-1]:.1f}}}\n')

# ----------------------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(F / f'{name}.pdf', bbox_inches='tight'); fig.savefig(F / f'{name}.png', dpi=300, bbox_inches='tight'); plt.close(fig)

def scatter_paired(ax, xb, xh, prot, title):
    for pr in 'AB':
        s = (prot == pr).values
        ax.scatter(xb[s], xh[s], s=26, marker=MRK[pr], color=COL[pr], edgecolor='white', linewidth=0.8, label=f'Protocol {pr}', zorder=3)
    ax.plot([0, 1], [0, 1], color='#52514e', lw=1, ls='--', zorder=1)
    ax.set_xlim(-0.03, 0.9); ax.set_ylim(-0.03, 0.9); ax.set_aspect('equal')
    ax.set_xlabel('Dice, baseline (FLAIR + T1)'); ax.set_ylabel('Dice, hybrid (+ anomaly map)'); ax.set_title(title, loc='left', fontsize=9)
    ax.text(0.86, 0.04, 'below the line:\nbaseline better', ha='right', va='bottom', fontsize=7, color='#52514e')

# Figure 3: CV and test paired Dice
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5))
scatter_paired(ax[0], B.dice.values, H.dice.values, B.protocole, f'(a) 5-fold CV, out-of-fold (n = 50), p = {pv(cv["p"])}')
scatter_paired(ax[1], T.dice_baseline.values, T.dice_hybride.values, T.protocole, f'(b) Held-out test (n = 27), p = {pv(tt["p"])}')
ax[0].legend(loc='upper left')
fig.tight_layout(); save(fig, 'fig3_paired_dice')

# Figure 4: fold-wise differences + ablation
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={'width_ratios': [1, 1.15]})
x = folds.index.values
ax[0].bar(x, folds.d, width=0.55, color='#9a9994', zorder=3)
ax[0].axhline(0, color='#0b0b0b', lw=0.8, zorder=5)
ax[0].axhline(cv['mean'], color='#0b0b0b', lw=1, ls='--', zorder=4)
ax[0].axhline(tt['mean'], color=COL['B'], lw=1.6, ls=':', zorder=4)
ax[0].text(-0.45, cv['mean'] - 0.012, f'dashed: CV mean ({cv["mean"]:+.3f})\ndotted: test mean ({tt["mean"]:+.3f})', va='top', ha='left', fontsize=7,
           bbox=dict(facecolor='white', edgecolor='none', pad=1), zorder=6)
ax[0].set_xticks(x); ax[0].set_xlabel('Fold'); ax[0].set_ylabel('Mean Dice difference (hybrid $-$ baseline)')
ax[0].set_title(f'(a) Hybrid better in {int((folds.d > 0).sum())}/5 folds', loc='left', fontsize=9)
cats = ['Baseline', 'Hybrid', 'Hybrid,\nmap = 0']
vals_cv = [cm['b'], cm['h'], cm['z']]; vals_t = [tm['b'], tm['h'], tm['z']]
xx = np.arange(3); w = 0.36
b1 = ax[1].bar(xx - w / 2 - 0.01, vals_cv, w, color=COL['A'], edgecolor='white', linewidth=2, label='5-fold CV (n = 50)', zorder=3)
b2 = ax[1].bar(xx + w / 2 + 0.01, vals_t, w, color='#9a9994', edgecolor='white', linewidth=2, hatch='///', label='Held-out test (n = 27)', zorder=3)
for bars in (b1, b2):
    for r_ in bars: ax[1].text(r_.get_x() + r_.get_width() / 2, r_.get_height() + 0.008, f'{r_.get_height():.3f}', ha='center', fontsize=7)
ax[1].set_xticks(xx); ax[1].set_xticklabels(cats); ax[1].set_ylabel('Mean Dice'); ax[1].set_ylim(0, 0.52)
ax[1].legend(loc='upper right'); ax[1].set_title('(b) Map reliance: map replaced by zeros', loc='left', fontsize=9)
fig.tight_layout(); save(fig, 'fig4_folds_ablation')

# Figure 5: map quality and H2
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
pos = {('entrainement', 'A'): 0, ('test', 'A'): 1, ('entrainement', 'B'): 2.6, ('test', 'B'): 3.6}
rs = np.random.RandomState(0)
for (sp, pr), p0 in pos.items():
    v = Q.loc[(Q.split == sp) & (Q.protocole == pr), 'map_auprc'].values
    ax[0].boxplot(v, positions=[p0], widths=0.55, showfliers=False, medianprops=dict(color='#0b0b0b'),
                  boxprops=dict(color='#52514e'), whiskerprops=dict(color='#52514e'), capprops=dict(color='#52514e'))
    ax[0].scatter(p0 + rs.uniform(-0.17, 0.17, len(v)), v, s=14, marker=MRK[pr], color=COL[pr], edgecolor='white', linewidth=0.5, zorder=3)
ax[0].set_xticks(list(pos.values())); ax[0].set_xticklabels(['A\ndevelopment', 'A\ntest', 'B\ndevelopment', 'B\ntest'])
ax[0].set_yscale('log'); ax[0].set_ylabel('AUPRC of the anomaly map alone (log)')
ax[0].set_title(f'(a) Map quality (development vs test p = {pv(R["q_mw_split"])})', loc='left', fontsize=9)
dl = H.dice - B.dice
for pr in 'AB':
    s = B.protocole == pr
    ax[1].scatter(Q.loc[B.index[s], 'map_auprc'], dl[s], s=24, marker=MRK[pr], color=COL[pr], edgecolor='white', linewidth=0.8, label=f'Protocol {pr}', zorder=3)
ax[1].axhline(0, color='#0b0b0b', lw=0.8); ax[1].set_xscale('log')
ax[1].set_xlabel('AUPRC of the anomaly map alone (log)'); ax[1].set_ylabel('Dice gain, hybrid $-$ baseline')
ax[1].set_title(f'(b) H2: Spearman $\\rho$ = {0.0 if abs(h2["rho"]) < 0.005 else h2["rho"]:.2f}, p = {pv(h2["p"])}', loc='left', fontsize=9)
ax[1].legend(loc='lower right')
fig.tight_layout(); save(fig, 'fig5_map_quality_H2')
print(json.dumps({k: M[k] for k in ['CVbase', 'CVhyb', 'CVdiff', 'CVlo', 'CVhi', 'CVp', 'Tbase', 'Thyb', 'Tdiff', 'Tp', 'CVzero', 'CVablP', 'HtwoRho', 'HtwoP']}, indent=0))
print('OK : numbers.tex, tables, figures written.')
