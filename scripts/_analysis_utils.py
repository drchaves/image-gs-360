"""Utilitários de análise (BD-Rate com PSNR, intervalos por bootstrap sobre imagens, tabelas LaTeX)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc_bd_rate_context import bjontegaard_rate  # noqa: E402  (mesma função usada nos demais resultados)

CAT_PT = {"anime": "Anime", "art": "Arte", "painting": "Pintura", "photo": "Fotografia", "vector": "Vetorial"}


def category(image_name):
    return image_name.split('-')[0]


def fnum(x, d=2, sign=False):
    """Número com vírgula decimal. Com sign=True devolve em modo matemático ($-0{,}50$), como o restante da tese."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if sign:
        return "$" + f"{x:+.{d}f}".replace('.', '{,}') + "$"
    return f"{x:.{d}f}".replace('.', ',')


def pm(x, d=2):
    """Percentual com sinal em modo matemático: $-1{,}48\\%$."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return "$" + f"{x:+.{d}f}".replace('.', '{,}') + "\\%$"


def complete_images(df, group_col, expected_values, extra_filter=None):
    """Imagens que possuem todos os valores esperados de group_col (após o filtro)."""
    d = df if extra_filter is None else df[extra_filter(df)]
    expected = set(expected_values)
    ok = []
    for img, g in d.groupby('image_name'):
        if set(g[group_col].round(6).tolist()) >= {round(float(v), 6) for v in expected}:
            ok.append(img)
    return sorted(ok)


def to_matrix(df, images, group_col, groups, rate='bpp', dist='psnr'):
    """Matrizes (n_imagens x n_grupos) de taxa e PSNR, na ordem de `groups`."""
    d = df.copy()
    d[group_col] = d[group_col].round(6)
    R = np.full((len(images), len(groups)), np.nan)
    P = np.full((len(images), len(groups)), np.nan)
    idx = {g: j for j, g in enumerate([round(float(x), 6) for x in groups])}
    pos = {im: i for i, im in enumerate(images)}
    for _, r in d[d['image_name'].isin(images)].iterrows():
        if r[group_col] in idx:
            R[pos[r['image_name']], idx[r[group_col]]] = r[rate]
            P[pos[r['image_name']], idx[r[group_col]]] = r[dist]
    return R, P


def pareto(rate, psnr):
    """Fronteira de Pareto (menor taxa, maior PSNR); devolve arrays ordenados por PSNR."""
    keep = []
    for i in range(len(rate)):
        dominated = any((rate[j] <= rate[i] and psnr[j] >= psnr[i]) and (rate[j] < rate[i] or psnr[j] > psnr[i])
                        for j in range(len(rate)))
        if not dominated:
            keep.append(i)
    keep = sorted(keep, key=lambda i: psnr[i])
    return np.array(rate)[keep], np.array(psnr)[keep]


def bd_from_matrices(Rref, Pref, Rtest, Ptest, rows=None, envelope=False):
    """BD-Rate (%) entre as curvas médias (sobre as linhas `rows`) de ref e test."""
    rows = slice(None) if rows is None else rows
    rr, pr = Rref[rows].mean(0), Pref[rows].mean(0)
    rt, pt = Rtest[rows].mean(0), Ptest[rows].mean(0)
    if envelope:
        rt, pt = pareto(rt, pt)
    if len(rt) < 2 or len(rr) < 2:
        return float('nan')
    return bjontegaard_rate(rr, pr, rt, pt)


def bootstrap_bd(Rref, Pref, Rtest, Ptest, n_boot=1000, seed=0, envelope=False):
    """(estimativa pontual, IC95% inferior, superior) do BD-Rate, reamostrando imagens com reposição."""
    n = Rref.shape[0]
    point = bd_from_matrices(Rref, Pref, Rtest, Ptest, envelope=envelope)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = bd_from_matrices(Rref, Pref, Rtest, Ptest, rows=idx, envelope=envelope)
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 10:
        return point, float('nan'), float('nan')
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def psnr_overlap(Pa, Pb):
    """Faixa de PSNR (dB) comum às curvas médias; devolve (min, max, largura)."""
    a, b = Pa.mean(0), Pb.mean(0)
    lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
    return lo, hi, hi - lo


def per_image_bd(Rref, Pref, Rtest, Ptest):
    out = []
    for i in range(Rref.shape[0]):
        out.append(bjontegaard_rate(Rref[i], Pref[i], Rtest[i], Ptest[i]))
    return np.array(out, dtype=float)

