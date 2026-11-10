"""Análise completa das 3 etapas finais:
  - Etapa 1: Ablação de Contexto C in {1, 2, 3, 4, 8, 16, 32} (Shallow, estabilizado, N=100k, 45 imagens)
  - Etapa 2: Validação de Arquitetura {'original', 'shallow', 'deep'} fixando C*=2 (N=100k, 45 imagens)
  - Etapa 3: Comparação Final: Image-GS original (baseline) vs. GaussianARM Vencedor (Shallow, C=2, estabilizado)
             em N in {10k, 30k, 50k, 100k} x 3 lambdas (45 imagens).

Entrada : resultados_diego_csvs/results_final_comparison.csv  (gerado por run_final_comparison.py)
          resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv (etapas intermediárias, N=100k)
Saída   : resultados_finais_c2/*.pdf|png, resultados_finais_c2/tabelas_final.tex e resumo_resultados_finais.md

Uso: python scripts/calc_final_comparison.py [--csv CAMINHO] [--archs-csv CAMINHO] [--out DIR] [--boot 1000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _analysis_utils import (CAT_PT, category, fnum, pm, complete_images, to_matrix,  # noqa: E402
                             bd_from_matrices, bootstrap_bd, psnr_overlap, per_image_bd)
from calc_bd_rate_context import bjontegaard_rate  # noqa: E402

N_LIST = [10000, 30000, 50000, 100000]
LAMBDAS = [0.0005, 0.0025, 0.0075]
CONTEXTS = [1, 2, 3, 4, 8, 16, 32]
ARCHS = ["original", "shallow", "deep"]
LAM_COLOR = {0.0005: "#1f77b4", 0.0025: "#2ca02c", 0.0075: "#d62728"}
METHOD_CONFIG = "shallow_c2_stab"
BASELINE_CONFIG = "image_gs_14bits"


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42,
    })
    sns.set_theme(style="whitegrid", rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
    })


def nlabel(n):
    return f"{n // 1000}k"


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def load(args):
    df = pd.read_csv(args.csv)
    for c in ['num_gaussians', 'lambda_rdo', 'psnr', 'bpp', 'bpp_img', 'bpp_arm', 'msssim', 'lpips']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    base = df[df['config'] == BASELINE_CONFIG].copy()
    ours = df[df['config'] == METHOD_CONFIG].copy()
    return df, base, ours


def plot_and_analyze_stage1_contexts(df, out_dir):
    """Analisa e plota Etapa 1: C in {1, 2, 3, 4, 8, 16, 32} com Shallow (N=100k)."""
    ctx_cfgs = {c: f"shallow_c{c}_stab" for c in CONTEXTS}
    sub = df[(df['num_gaussians'] == 100000) & (df['config'].isin(ctx_cfgs.values()))].copy()
    sub['arm_context'] = sub['arm_context'].astype(int)
    sub['cat'] = sub['image_name'].apply(category)

    imgs = sorted(sub['image_name'].unique())
    mean_df = sub.groupby(['arm_context', 'lambda_rdo'], as_index=False)[['bpp', 'bpp_img', 'bpp_arm', 'psnr', 'msssim', 'lpips']].mean()

    anc_tot = mean_df[mean_df['arm_context'] == 16].sort_values('psnr')
    r_anc_tot, p_anc = anc_tot['bpp'].values, anc_tot['psnr'].values
    r_anc_img = anc_tot['bpp_img'].values

    stage1_stats = []
    for c in CONTEXTS:
        sc = mean_df[mean_df['arm_context'] == c].sort_values('psnr')
        bd_tot = 0.0 if c == 16 else bjontegaard_rate(r_anc_tot, p_anc, sc['bpp'].values, sc['psnr'].values)
        bd_img = 0.0 if c == 16 else bjontegaard_rate(r_anc_img, p_anc, sc['bpp_img'].values, sc['psnr'].values)
        stage1_stats.append((
            c, sc['bpp_arm'].mean(), sc['bpp_img'].mean(), sc['bpp'].mean(),
            sc['psnr'].mean(), sc['msssim'].mean(), sc['lpips'].mean(), bd_tot, bd_img
        ))

    ctx_colors = {
        1:  "#ff7f0e",  # Laranja
        2:  "#d62728",  # Vermelho destaque (Campeão)
        3:  "#e377c2",  # Rosa
        4:  "#1f77b4",  # Azul
        8:  "#2ca02c",  # Verde
        16: "#333333",  # Preto/Cinza escuro (Âncora)
        32: "#9467bd",  # Roxo
    }
    ctx_markers = {1: "v", 2: "*", 3: "P", 4: "o", 8: "s", 16: "D", 32: "^"}
    ctx_ls = {1: "--", 2: "-", 3: "-.", 4: "--", 8: "-.", 16: "-", 32: ":"}

    # 1A) Lado a lado: BPP Total vs BPP Imagem
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)
    for ax, (rcol, t_suf, x_lbl) in zip(axes, [
        ('bpp', 'Taxa Total (Imagem + Pesos do ARM)', 'Taxa Total — Bits por Pixel (bpp)'),
        ('bpp_img', 'Apenas Imagem (Sem Overhead do ARM)', 'Taxa das Gaussianas — bpp (bpp_img)')
    ]):
        anc_s = mean_df[mean_df['arm_context'] == 16].sort_values('psnr')
        for c in CONTEXTS:
            s = mean_df[mean_df['arm_context'] == c].sort_values('lambda_rdo', ascending=False)
            sp = s.sort_values('psnr')
            bd_v = 0.0 if c == 16 else bjontegaard_rate(anc_s[rcol].values, anc_s['psnr'].values, sp[rcol].values, sp['psnr'].values)
            bd_txt = "Âncora" if c == 16 else f"BD: {bd_v:+.2f}%"
            arm_v = s['bpp_arm'].iloc[0]
            lbl = f"C={c} (arm={arm_v:.4f}) [{bd_txt}]"
            ax.plot(
                s[rcol], s['psnr'],
                linestyle=ctx_ls[c], marker=ctx_markers[c], color=ctx_colors[c],
                linewidth=2.8 if c == 2 else (2.2 if c == 16 else 1.8),
                markersize=10 if c == 2 else 7,
                label=lbl
            )
        ax.set_title(f"PSNR vs. bpp — {t_suf}", fontsize=12.5, fontweight="bold")
        ax.set_xlabel(x_lbl, fontsize=11)
        ax.grid(True, ls="--", alpha=0.6)
        ax.legend(fontsize=8.8, loc="lower right", frameon=True)
    axes[0].set_ylabel("PSNR Médio (dB)", fontsize=11)
    fig.suptitle(f"Etapa 1: Ablação de Contexto $C \\in \\{{1, 2, 3, 4, 8, 16, 32\\}}$ (Shallow, N=100k — {len(imgs)} imagens)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "etapa1_contextos_total_vs_img")

    # 1B) Painel 2x3 por Categoria
    cats = sorted(sub['cat'].unique())
    fig_c, axes_c = plt.subplots(2, 3, figsize=(18, 10.8))
    axes_c = axes_c.flatten()
    for idx, cat in enumerate(cats + ['__global__']):
        ax = axes_c[idx]
        d_sub = mean_df if cat == '__global__' else sub[sub['cat'] == cat].groupby(['arm_context', 'lambda_rdo'], as_index=False)[['bpp', 'psnr']].mean()
        anc_c = d_sub[d_sub['arm_context'] == 16].sort_values('psnr')
        for c in CONTEXTS:
            s = d_sub[d_sub['arm_context'] == c].sort_values('lambda_rdo', ascending=False)
            sp = s.sort_values('psnr')
            bd_v = 0.0 if c == 16 else bjontegaard_rate(anc_c['bpp'].values, anc_c['psnr'].values, sp['bpp'].values, sp['psnr'].values)
            bd_txt = "Âncora" if c == 16 else f"BD: {bd_v:+.2f}%"
            ax.plot(
                s['bpp'], s['psnr'],
                linestyle=ctx_ls[c], marker=ctx_markers[c], color=ctx_colors[c],
                linewidth=2.6 if c == 2 else 1.8, markersize=9 if c == 2 else 6.5,
                label=f"C={c} [{bd_txt}]"
            )
        ttl = f"Média Global ({len(imgs)} imagens)" if cat == '__global__' else f"Categoria: {CAT_PT.get(cat, cat)} (9 imagens)"
        ax.set_title(ttl, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Taxa Total — bpp", fontsize=10.5)
        ax.set_ylabel("PSNR (dB)", fontsize=10.5)
        ax.legend(fontsize=8.2, loc="lower right", frameon=True)
        ax.grid(True, ls="--", alpha=0.6)
    fig_c.suptitle("Etapa 1: Curvas Taxa-Distorção por Categoria — Contextos $C \\in \\{1, 2, 3, 4, 8, 16, 32\\}$", fontsize=15, fontweight="bold", y=1.01)
    fig_c.tight_layout()
    save(fig_c, out_dir, "etapa1_contextos_por_categoria")

    return stage1_stats


def plot_and_analyze_stage2_archs(df, out_dir):
    """Analisa e plota Etapa 2: Arquiteturas {'original', 'shallow', 'deep'} fixando C*=2 (N=100k)."""
    arch_cfgs = {a: f"{a}_c2_stab" for a in ARCHS}
    sub = df[(df['num_gaussians'] == 100000) & (df['config'].isin(arch_cfgs.values()))].copy()
    sub['cat'] = sub['image_name'].apply(category)
    imgs = sorted(sub['image_name'].unique())
    mean_df = sub.groupby(['arm_arch', 'lambda_rdo'], as_index=False)[['bpp', 'bpp_img', 'bpp_arm', 'psnr', 'msssim', 'lpips']].mean()

    anc_tot = mean_df[mean_df['arm_arch'] == 'original'].sort_values('psnr')
    r_anc_tot, p_anc = anc_tot['bpp'].values, anc_tot['psnr'].values
    r_anc_img = anc_tot['bpp_img'].values

    stage2_stats = []
    for a in ARCHS:
        sa = mean_df[mean_df['arm_arch'] == a].sort_values('psnr')
        bd_tot = 0.0 if a == 'original' else bjontegaard_rate(r_anc_tot, p_anc, sa['bpp'].values, sa['psnr'].values)
        bd_img = 0.0 if a == 'original' else bjontegaard_rate(r_anc_img, p_anc, sa['bpp_img'].values, sa['psnr'].values)
        stage2_stats.append((
            a, sa['bpp_arm'].mean(), sa['bpp_img'].mean(), sa['bpp'].mean(),
            sa['psnr'].mean(), sa['msssim'].mean(), sa['lpips'].mean(), bd_tot, bd_img
        ))

    arch_colors = {"original": "#1b7a5a", "shallow": "#d62728", "deep": "#5a4bb5"}
    arch_markers = {"original": "o", "shallow": "s", "deep": "^"}
    arch_ls = {"original": "--", "shallow": "-", "deep": "-."}

    # 2A) Lado a lado: BPP Total vs BPP Imagem em C*=2
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0), sharey=True)
    for ax, (rcol, t_suf, x_lbl) in zip(axes, [
        ('bpp', 'Taxa Total (Imagem + Pesos do ARM)', 'Taxa Total — Bits por Pixel (bpp)'),
        ('bpp_img', 'Apenas Imagem (Sem Overhead do ARM)', 'Taxa das Gaussianas — bpp (bpp_img)')
    ]):
        anc_s = mean_df[mean_df['arm_arch'] == 'original'].sort_values('psnr')
        for a in ARCHS:
            s = mean_df[mean_df['arm_arch'] == a].sort_values('lambda_rdo', ascending=False)
            sp = s.sort_values('psnr')
            bd_v = 0.0 if a == 'original' else bjontegaard_rate(anc_s[rcol].values, anc_s['psnr'].values, sp[rcol].values, sp['psnr'].values)
            bd_txt = "Âncora" if a == 'original' else f"BD: {bd_v:+.2f}%"
            arm_v = s['bpp_arm'].iloc[0]
            ax.plot(
                s[rcol], s['psnr'],
                linestyle=arch_ls[a], marker=arch_markers[a], color=arch_colors[a],
                linewidth=2.6 if a == 'shallow' else 2.0, markersize=8.5 if a == 'shallow' else 7,
                label=f"{a.capitalize()} C=2 (arm={arm_v:.4f}) [{bd_txt}]"
            )
        ax.set_title(f"PSNR vs. bpp — {t_suf}", fontsize=12.5, fontweight="bold")
        ax.set_xlabel(x_lbl, fontsize=11)
        ax.grid(True, ls="--", alpha=0.6)
        ax.legend(fontsize=9.5, loc="lower right", frameon=True)
    axes[0].set_ylabel("PSNR Médio (dB)", fontsize=11)
    fig.suptitle(f"Etapa 2: Validação das Arquiteturas Fixando C*=2 (N=100k — {len(imgs)} imagens)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "etapa2_arquiteturas_c2_total_vs_img")

    # 2B) Painel 2x3 por Categoria
    cats = sorted(sub['cat'].unique())
    fig_c, axes_c = plt.subplots(2, 3, figsize=(18, 10.5))
    axes_c = axes_c.flatten()
    for idx, cat in enumerate(cats + ['__global__']):
        ax = axes_c[idx]
        d_sub = mean_df if cat == '__global__' else sub[sub['cat'] == cat].groupby(['arm_arch', 'lambda_rdo'], as_index=False)[['bpp', 'psnr']].mean()
        anc_c = d_sub[d_sub['arm_arch'] == 'original'].sort_values('psnr')
        for a in ARCHS:
            s = d_sub[d_sub['arm_arch'] == a].sort_values('lambda_rdo', ascending=False)
            sp = s.sort_values('psnr')
            bd_v = 0.0 if a == 'original' else bjontegaard_rate(anc_c['bpp'].values, anc_c['psnr'].values, sp['bpp'].values, sp['psnr'].values)
            bd_txt = "Âncora" if a == 'original' else f"BD: {bd_v:+.2f}%"
            ax.plot(
                s['bpp'], s['psnr'],
                linestyle=arch_ls[a], marker=arch_markers[a], color=arch_colors[a],
                linewidth=2.5 if a == 'shallow' else 1.9, markersize=8 if a == 'shallow' else 6.8,
                label=f"{a.capitalize()} (C=2) [{bd_txt}]"
            )
        ttl = f"Média Global ({len(imgs)} imagens)" if cat == '__global__' else f"Categoria: {CAT_PT.get(cat, cat)} (9 imagens)"
        ax.set_title(ttl, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Taxa Total — bpp", fontsize=10.5)
        ax.set_ylabel("PSNR (dB)", fontsize=10.5)
        ax.legend(fontsize=9.2, loc="lower right", frameon=True)
        ax.grid(True, ls="--", alpha=0.6)
    fig_c.suptitle("Etapa 2: Arquiteturas do ARM (Original, Shallow, Deep) Fixando C*=2 por Categoria", fontsize=15, fontweight="bold", y=1.01)
    fig_c.tight_layout()
    save(fig_c, out_dir, "etapa2_arquiteturas_c2_por_categoria")

    return stage2_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default="resultados_diego_csvs/results_final_comparison.csv")
    ap.add_argument('--archs-csv', default="resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv")
    ap.add_argument('--out', default="resultados_finais_c2")
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_style()

    df, base, ours = load(args)
    stage1_stats = plot_and_analyze_stage1_contexts(df, args.out)
    stage2_stats = plot_and_analyze_stage2_archs(df, args.out)

    imgs_b = set(complete_images(base, 'num_gaussians', N_LIST))
    imgs_o = set(complete_images(ours, 'num_gaussians', N_LIST))
    # exige as 3 lambdas completas para cada N
    imgs_o = {i for i in imgs_o
              if all(len(ours[(ours.image_name == i) & (np.isclose(ours.lambda_rdo, l))]) == len(N_LIST) for l in LAMBDAS)}
    imgs = sorted(imgs_b & imgs_o)
    n_img = len(imgs)
    print(f"Imagens no baseline (completas): {len(imgs_b)} | no método (completas): {len(imgs_o)} | em ambos: {n_img}")
    if n_img == 0:
        print("Nenhuma imagem completa nos dois lados ainda. Rode run_final_comparison.py.")
        return
    missing = sorted((imgs_b | imgs_o) - set(imgs))
    if missing:
        print("Imagens fora da análise (incompletas em algum lado):", missing)

    Rb, Pb = to_matrix(base, imgs, 'num_gaussians', N_LIST)
    mats = {}
    for lam in LAMBDAS:
        sub = ours[np.isclose(ours.lambda_rdo, lam)]
        mats[lam] = to_matrix(sub, imgs, 'num_gaussians', N_LIST)
    Rall = np.concatenate([mats[l][0] for l in LAMBDAS], axis=1)
    Pall = np.concatenate([mats[l][1] for l in LAMBDAS], axis=1)

    # ------------------------------------------------------------------ 1. BD-Rate global (IC95%)
    print("\n=== BD-Rate (PSNR) vs. baseline — curva média global, IC95% por bootstrap sobre imagens ===")
    bd_glob = {}
    for lam in LAMBDAS:
        R, P = mats[lam]
        pt, lo, hi = bootstrap_bd(Rb, Pb, R, P, n_boot=args.boot)
        a, b, w = psnr_overlap(Pb, P)
        bd_glob[lam] = (pt, lo, hi)
        flag = "  [ATENÇÃO: sobreposição de PSNR < 3 dB]" if w < 3 else ""
        print(f"  lambda={lam:<7g}: {pt:+6.2f}%  IC95% [{lo:+.2f}; {hi:+.2f}] | PSNR comum {a:.1f}-{b:.1f} dB ({w:.1f} dB){flag}")
    pt, lo, hi = bootstrap_bd(Rb, Pb, Rall, Pall, n_boot=args.boot, envelope=True)
    bd_glob['env'] = (pt, lo, hi)
    print(f"  envelope      : {pt:+6.2f}%  IC95% [{lo:+.2f}; {hi:+.2f}]")

    # ------------------------------------------------------------------ 2. BD-Rate por categoria
    print("\n=== BD-Rate (PSNR) por categoria ===")
    cats = sorted({category(i) for i in imgs})
    bd_cat = {}
    for cat in cats:
        rows = [k for k, i in enumerate(imgs) if category(i) == cat]
        vals = {}
        for lam in LAMBDAS:
            R, P = mats[lam]
            vals[lam] = bd_from_matrices(Rb, Pb, R, P, rows=rows)
        vals['env'] = bd_from_matrices(Rb, Pb, Rall, Pall, rows=rows, envelope=True)
        bd_cat[cat] = vals
        print(f"  {cat:9s}({len(rows)} imgs): " + " | ".join(f"{k if k=='env' else f'L={k:g}'}: {v:+.2f}%" for k, v in vals.items()))

    # ------------------------------------------------------------------ 3. BD-Rate por imagem
    print("\n=== BD-Rate por imagem (curvas sobre N) ===")
    for lam in LAMBDAS:
        R, P = mats[lam]
        v = per_image_bd(Rb, Pb, R, P)
        ok = v[~np.isnan(v)]
        print(f"  lambda={lam:<7g}: mediana {np.median(ok):+.2f}% | média {np.mean(ok):+.2f}% | "
              f"imagens com BD<0: {np.sum(ok < 0)}/{len(ok)} | válidas {len(ok)}/{len(v)}")

    # ------------------------------------------------------------------ 4. Comparação pareada por N
    print("\n=== Comparação pareada por N e lambda (médias sobre as imagens) ===")
    pair_rows = []
    for j, n in enumerate(N_LIST):
        bb, bp = Rb[:, j].mean(), Pb[:, j].mean()
        for lam in LAMBDAS:
            R, P = mats[lam]
            ob, op = R[:, j].mean(), P[:, j].mean()
            sub = ours[(ours.num_gaussians == n) & np.isclose(ours.lambda_rdo, lam) & ours.image_name.isin(imgs)]
            pair_rows.append((n, lam, bb, bp, ob, op, (ob / bb - 1) * 100, op - bp,
                              sub.bpp_img.mean(), sub.bpp_arm.mean()))
            print(f"  N={nlabel(n):>4s} L={lam:<7g}: base {bb:.4f} bpp / {bp:.2f} dB | método {ob:.4f} bpp / {op:.2f} dB "
                  f"| Δbpp {pair_rows[-1][6]:+.1f}% | ΔPSNR {pair_rows[-1][7]:+.2f} dB")

    # ------------------------------------------------------------------ 5. Progressão em N=100k
    print("\n=== Progressão das etapas em N=100k (curvas sobre lambda) ===")
    stages = []
    if os.path.isfile(args.archs_csv):
        ar = pd.read_csv(args.archs_csv)
        for c in ['lambda_rdo', 'psnr', 'bpp']:
            ar[c] = ar[c].astype(float)
        ar['stab'] = ar['stabilize_grads'].astype(str).str.strip().str.lower()
        for label, arch, stab in [("Original, $C=16$, sem estab.", 'original', 'false'),
                                  ("Original, $C=16$, estab.", 'original', 'true'),
                                  ("\\textit{Shallow}, $C=16$, estab.", 'shallow', 'true')]:
            stages.append((label, ar[(ar.arm_arch == arch) & (ar.stab == stab)]))
    c4_100 = df[(df.config == 'shallow_c4_stab') & (df.num_gaussians == 100000)]
    if not c4_100.empty:
        stages.append(("\\textit{Shallow}, $C=4$, estab.", c4_100))
    ours100 = ours[ours.num_gaussians == 100000]
    stages.append(("\\textit{Shallow}, $C=2$, estab. (Final)", ours100))

    stage_imgs = set(imgs)
    for _, d in stages:
        stage_imgs &= set(complete_images(d, 'lambda_rdo', LAMBDAS))
    stage_imgs = sorted(stage_imgs)
    print(f"  imagens comuns a todas as etapas: {len(stage_imgs)}")
    prog_rows, prev = [], None
    rows_b = [imgs.index(i) for i in stage_imgs]
    prog_mats = []
    for label, d in stages:
        R, P = to_matrix(d, stage_imgs, 'lambda_rdo', LAMBDAS)
        prog_mats.append((label, R, P))
        bd_b = bjb(Rb[rows_b], Pb[rows_b], R, P)
        bd_prev = bd_from_matrices(prev[1], prev[2], R, P) if prev is not None else float('nan')
        prog_rows.append((label, R.mean(), P.mean(), bd_b, bd_prev))
        print(f"  {label.replace('$','').replace(chr(92)+'textit{','').replace('}','')[:36]:36s}: bpp {R.mean():.4f} | PSNR {P.mean():.2f} dB | "
              f"BD vs baseline {bd_b:+.2f}% | BD vs etapa anterior {bd_prev:+.2f}%")
        prev = (label, R, P)

    # ------------------------------------------------------------------ Gráficos da Etapa 3
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.plot(Rb.mean(0), Pb.mean(0), 'k-s', lw=2.2, ms=8, label="Image-GS original (baseline)")
    for j, n in enumerate(N_LIST):
        ax.annotate(f"N={nlabel(n)}", (Rb[:, j].mean(), Pb[:, j].mean()), textcoords="offset points", xytext=(6, -12), fontsize=8.5)
    for lam in LAMBDAS:
        R, P = mats[lam]
        b = bd_glob[lam][0]
        bd_lbl = f"BD: {b:+.2f}%" if not np.isnan(b) else "alta compressão"
        ax.plot(R.mean(0), P.mean(0), '-o', color=LAM_COLOR[lam], lw=2.2, ms=7,
                label=f"GaussianARM (Shallow, C=2), $\\lambda$={lam:g}  [{bd_lbl}]")
    ax.set_xlabel("Taxa total — bpp"); ax.set_ylabel("PSNR médio (dB)")
    ax.set_title(f"Etapa 3: Taxa-distorção Final (PSNR vs. bpp) — {n_img} imagens", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.6); ax.legend(fontsize=9, loc="lower right")
    save(fig, args.out, "etapa3_rd_final_global")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5)); axes = axes.flatten()
    for k, cat in enumerate(cats + ['__global__']):
        ax = axes[k]
        rows = list(range(n_img)) if cat == '__global__' else [i for i, im in enumerate(imgs) if category(im) == cat]
        ax.plot(Rb[rows].mean(0), Pb[rows].mean(0), 'k-s', lw=2.2, ms=7, label="Image-GS (baseline)")
        for lam in LAMBDAS:
            R, P = mats[lam]
            b = bd_glob[lam][0] if cat == '__global__' else bd_cat[cat][lam]
            bd_lbl = f"BD: {b:+.2f}%" if not np.isnan(b) else "BD: N/A"
            ax.plot(R[rows].mean(0), P[rows].mean(0), '-o', color=LAM_COLOR[lam], lw=2.0, ms=6, label=f"$\\lambda$={lam:g} [{bd_lbl}]")
        title = f"Média global ({n_img} imagens)" if cat == '__global__' else f"Categoria: {CAT_PT.get(cat, cat)} ({len(rows)} imagens)"
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Taxa total — bpp"); ax.set_ylabel("PSNR (dB)")
        ax.grid(True, ls="--", alpha=0.6); ax.legend(fontsize=8.5, loc="lower right")
    for k in range(len(cats) + 1, 6):
        axes[k].axis('off')
    fig.suptitle("Etapa 3: Image-GS original vs. GaussianARM (Shallow, C=2) — PSNR vs. bpp por categoria", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, args.out, "etapa3_rd_final_por_categoria")

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.plot(Rb.mean(0), Pb.mean(0), 'k-s', lw=2.2, ms=8, label="Image-GS original (curva sobre N)")
    colors = ["#8c564b", "#ff7f0e", "#9467bd", "#1f77b4", "#d62728"]
    for (label, R, P), col in zip(prog_mats, colors):
        lbl = label.replace("\\textit{", "").replace("}", "").replace("$", "")
        ax.plot(R.mean(0), P.mean(0), '-o', color=col, lw=2.2 if "C=2" in lbl else 1.8, ms=7 if "C=2" in lbl else 6, label=f"{lbl} (N=100k, 3 $\\lambda$)")
    ax.set_xlabel("Taxa total — bpp"); ax.set_ylabel("PSNR médio (dB)")
    ax.set_title(f"Progressão das etapas em N=100k — {len(stage_imgs)} imagens", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.6); ax.legend(fontsize=8.5, loc="lower right")
    save(fig, args.out, "etapa3_rd_final_progressao")

    # ------------------------------------------------------------------ Tabelas LaTeX (estilo da tese)
    t = []
    t.append("% ===== Tabela Etapa 1: Ablação de Contexto C in {1, 2, 3, 4, 8, 16, 32} =====")
    t.append("\\begin{table}[htbp]\n    \\centering")
    t.append("    \\caption{Etapa 1: Ablação do tamanho de contexto $C \\in \\{1, 2, 3, 4, 8, 16, 32\\}$ com arquitetura \\textit{Shallow} estabilizada ($N=100.000$, médias nas 45 imagens).}")
    t.append("    \\label{tab:etapa1_contextos}\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{|c|c|c|c|c|c|c|c|c|}\n        \\hline")
    t.append("        $\\boldsymbol{C}$ & \\textbf{bpp\\_arm ($\\downarrow$)} & \\textbf{bpp\\_img ($\\downarrow$)} & \\textbf{bpp Total ($\\downarrow$)} & \\textbf{PSNR (dB) ($\\uparrow$)} & \\textbf{MS-SSIM ($\\uparrow$)} & \\textbf{LPIPS ($\\downarrow$)} & \\textbf{BD-Rate (bpp Total)} & \\textbf{BD-Rate (Só bpp\\_img)} \\\\ \\hline\\hline")
    for c, ba, bi, bt, ps, ms, lp, bdt, bdi in stage1_stats:
        bdt_s = "Âncora" if c == 16 else pm(bdt)
        bdi_s = "Âncora" if c == 16 else pm(bdi)
        t.append(f"        {c} & {fnum(ba, 4)} & {fnum(bi, 4)} & {fnum(bt, 4)} & {fnum(ps, 2)} & {fnum(ms, 4)} & {fnum(lp, 4)} & {bdt_s} & {bdi_s} \\\\ \\hline")
    t.append("    \\end{tabular}\n    }\n    \\fonte O autor (2026).\n\\end{table}\n")

    t.append("% ===== Tabela Etapa 2: Validação de Arquitetura fixando C*=2 =====")
    t.append("\\begin{table}[htbp]\n    \\centering")
    t.append("    \\caption{Etapa 2: Comparação entre as arquiteturas \\textit{Original}, \\textit{Shallow} e \\textit{Deep} fixando o melhor contexto $C^*=2$ ($N=100.000$, médias nas 45 imagens).}")
    t.append("    \\label{tab:etapa2_arquiteturas_c2}\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\n        \\hline")
    t.append("        \\textbf{Arquitetura ($C^*=2$)} & \\textbf{bpp\\_arm ($\\downarrow$)} & \\textbf{bpp\\_img ($\\downarrow$)} & \\textbf{bpp Total ($\\downarrow$)} & \\textbf{PSNR (dB) ($\\uparrow$)} & \\textbf{MS-SSIM ($\\uparrow$)} & \\textbf{LPIPS ($\\downarrow$)} & \\textbf{BD-Rate (bpp Total)} & \\textbf{BD-Rate (Só bpp\\_img)} \\\\ \\hline\\hline")
    for a, ba, bi, bt, ps, ms, lp, bdt, bdi in stage2_stats:
        bdt_s = "Âncora" if a == 'original' else pm(bdt)
        bdi_s = "Âncora" if a == 'original' else pm(bdi)
        t.append(f"        \\textit{{{a.capitalize()}}} ($C=2$) & {fnum(ba, 4)} & {fnum(bi, 4)} & {fnum(bt, 4)} & {fnum(ps, 2)} & {fnum(ms, 4)} & {fnum(lp, 4)} & {bdt_s} & {bdi_s} \\\\ \\hline")
    t.append("    \\end{tabular}\n    }\n    \\fonte O autor (2026).\n\\end{table}\n")

    t.append("% ===== Tabela Etapa 3 (A): comparação pareada por N e lambda =====")
    t.append("\\begin{table}[htbp]\n    \\centering")
    t.append("    \\caption{Comparação entre o \\textit{Image-GS} original e o \\textit{GaussianARM} vencedor (\\textit{Shallow}, $C=2$, estabilizado), "
             "por número de gaussianas $N$ e $\\lambda$. Médias sobre " + str(n_img) + " imagens; o bpp do método inclui os pesos do ARM.}")
    t.append("    \\label{tab:final_pareada}\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{|c|c|cc|cc|c|c|}\n        \\hline")
    t.append("        \\multirow{2}{*}{$\\boldsymbol{N}$} & \\multirow{2}{*}{$\\boldsymbol{\\lambda}$} & \\multicolumn{2}{c|}{\\textbf{\\textit{Image-GS}}} "
             "& \\multicolumn{2}{c|}{\\textbf{\\textit{GaussianARM} (\\textit{Shallow}, $C=2$)}} & \\multirow{2}{*}{$\\boldsymbol{\\Delta}$\\textbf{bpp}} & \\multirow{2}{*}{$\\boldsymbol{\\Delta}$\\textbf{PSNR (dB)}} \\\\ \\cline{3-6}")
    t.append("        & & \\textbf{bpp ($\\downarrow$)} & \\textbf{PSNR (dB) ($\\uparrow$)} & \\textbf{bpp ($\\downarrow$)} & \\textbf{PSNR (dB) ($\\uparrow$)} & & \\\\ \\hline\\hline")
    for n in N_LIST:
        rows = [r for r in pair_rows if r[0] == n]
        for k, r in enumerate(rows):
            first = ("\\multirow{3}{*}{" + f"{n:,}".replace(',', '.') + "}") if k == 0 else ""
            sep = "\\cline{2-8}" if k < 2 else "\\hline"
            t.append(f"        {first} & {fnum(r[1], 4)} & {fnum(r[2], 4)} & {fnum(r[3])} & {fnum(r[4], 4)} & {fnum(r[5])} & {pm(r[6], 1)} & {fnum(r[7], 2, True)} \\\\ {sep}")
    t.append("    \\end{tabular}\n    }\n    \\fonte O autor (2026).\n\\end{table}\n")

    t.append("% ===== Tabela Etapa 3 (B): BD-Rate (PSNR) em relação ao Image-GS original =====")
    t.append("\\begin{table}[htbp]\n    \\centering")
    t.append("    \\caption{BD-Rate (PSNR) do \\textit{GaussianARM} (\\textit{Shallow}, $C=2$) em relação ao \\textit{Image-GS} original. Cada coluna $\\lambda$ usa a curva sobre $N$ "
             "daquele $\\lambda$; \\emph{Envelope} usa a fronteira de Pareto dos 12 pontos. Entre colchetes, IC de 95\\% (\\textit{bootstrap} sobre imagens). Valores negativos indicam menor taxa para o mesmo PSNR.}")
    t.append("    \\label{tab:final_bdrate}\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{|l|c|c|c|c|}\n        \\hline")
    t.append("        \\textbf{Escopo} & $\\boldsymbol{\\lambda=0{,}0005}$ & $\\boldsymbol{\\lambda=0{,}0025}$ & $\\boldsymbol{\\lambda=0{,}0075}$ & \\textbf{Envelope} \\\\ \\hline\\hline")
    cells = []
    for key in LAMBDAS + ['env']:
        p_, lo_, hi_ = bd_glob[key]
        cells.append(f"{pm(p_)} [{fnum(lo_, 2, True)}; {fnum(hi_, 2, True)}]")
    t.append(f"        Global ({n_img} imagens) & " + " & ".join(cells) + " \\\\ \\hline")
    for cat in cats:
        t.append(f"        {CAT_PT.get(cat, cat)} & " + " & ".join(pm(bd_cat[cat][k]) for k in LAMBDAS + ['env']) + " \\\\ \\hline")
    t.append("    \\end{tabular}\n    }\n    \\fonte O autor (2026).\n\\end{table}\n")

    t.append("% ===== Tabela Etapa 3 (C): progressão das etapas em N=100.000 =====")
    t.append("\\begin{table}[htbp]\n    \\centering")
    t.append("    \\caption{Progressão das etapas em $N=100.000$ gaussianas (médias sobre os três $\\lambda$ e " + str(len(stage_imgs)) +
             " imagens). O BD-Rate é calculado sobre o PSNR em relação ao \\textit{Image-GS} original (curva sobre $N$) e à etapa imediatamente anterior.}")
    t.append("    \\label{tab:progressao_etapas}\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{|l|c|c|c|c|}\n        \\hline")
    t.append("        \\textbf{Etapa} & \\textbf{bpp ($\\downarrow$)} & \\textbf{PSNR (dB) ($\\uparrow$)} & \\textbf{BD-Rate vs. \\textit{Image-GS}} & \\textbf{BD-Rate vs. etapa anterior} \\\\ \\hline\\hline")
    for label, b_, p_, bdb, bdp in prog_rows:
        t.append(f"        {label} & {fnum(b_, 4)} & {fnum(p_)} & {pm(bdb)} & {pm(bdp) if not np.isnan(bdp) else '---'} \\\\ \\hline")
    t.append("    \\end{tabular}\n    }\n    \\fonte O autor (2026).\n\\end{table}\n")
    with open(os.path.join(args.out, "tabelas_final.tex"), "w") as f:
        f.write("\n".join(t))

    # Salva também um resumo legível em Markdown na mesma pasta
    md = [
        "# Resumo Consolidado dos Resultados Finais (`shallow_c2_stab`)\n",
        "## 1. Etapa 1: Ablação de Contexto ($C \\in \\{1, 2, 3, 4, 8, 16, 32\\}$, Arquitetura `shallow`, $N=100\\text{k}$, 45 imagens)\n",
        "| Contexto ($C$) | `bpp_arm` | `bpp_img` | `bpp` Total | PSNR (dB) | MS-SSIM | LPIPS | BD-Rate (`bpp` Total) | BD-Rate (Só `bpp_img`) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for c, ba, bi, bt, ps, ms, lp, bdt, bdi in stage1_stats:
        bdt_s = "0.00% (Âncora)" if c == 16 else f"{bdt:+.2f}%"
        bdi_s = "0.00% (Âncora)" if c == 16 else f"{bdi:+.2f}%"
        md.append(f"| **C={c}** | {ba:.4f} | {bi:.4f} | {bt:.4f} | {ps:.2f} | {ms:.4f} | {lp:.4f} | **{bdt_s}** | **{bdi_s}** |")

    md.extend([
        "\n## 2. Etapa 2: Validação de Arquitetura Fixando $C^* = 2$ ($N=100\\text{k}$, 45 imagens)\n",
        "| Arquitetura ($C^*=2$) | `bpp_arm` | `bpp_img` | `bpp` Total | PSNR (dB) | MS-SSIM | LPIPS | BD-Rate (`bpp` Total) | BD-Rate (Só `bpp_img`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for a, ba, bi, bt, ps, ms, lp, bdt, bdi in stage2_stats:
        bdt_s = "0.00% (Âncora)" if a == 'original' else f"{bdt:+.2f}%"
        bdi_s = "0.00% (Âncora)" if a == 'original' else f"{bdi:+.2f}%"
        md.append(f"| **`{a}_c2_stab`** | {ba:.4f} | {bi:.4f} | {bt:.4f} | {ps:.2f} | {ms:.4f} | {lp:.4f} | **{bdt_s}** | **{bdi_s}** |")

    md.extend([
        "\n## 3. Etapa 3: Comparação Pareada por $N$ e $\\lambda$ (`shallow_c2_stab` vs. `image_gs_14bits`, 45 imagens)\n",
        "| $N$ | $\\lambda$ | Baseline BPP | Baseline PSNR (dB) | `shallow_c2_stab` BPP | `shallow_c2_stab` PSNR (dB) | $\\Delta$ BPP (%) | $\\Delta$ PSNR (dB) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in pair_rows:
        md.append(f"| {r[0]} | {r[1]:g} | {r[2]:.4f} | {r[3]:.2f} | {r[4]:.4f} | {r[5]:.2f} | **{r[6]:+.2f}%** | {r[7]:+.2f} |")

    with open(os.path.join(args.out, "resumo_resultados_finais.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nTodos os gráficos, tabelas LaTeX e resumo foram salvos em '{args.out}/'!")


def bjb(Rref, Pref, Rtest, Ptest):
    from _analysis_utils import bjontegaard_rate
    return bjontegaard_rate(Rref.mean(0), Pref.mean(0), Rtest.mean(0), Ptest.mean(0))


if __name__ == "__main__":
    main()
