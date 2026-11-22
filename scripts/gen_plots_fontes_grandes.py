"""Gera todas as figuras de resultados do TCC com fontes ampliadas (legenda, eixos, ticks e títulos)
e salva com o sufixo `_fontes_grandes.{pdf,png}` em `tcc_diego/figuras/`.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _analysis_utils import (CAT_PT, category, complete_images, to_matrix,  # noqa: E402
                             bd_from_matrices, bootstrap_bd, per_image_bd)
from calc_bd_rate_context import bjontegaard_rate  # noqa: E402

OUT_DIR = "tcc_diego/figuras"
N_LIST = [10000, 30000, 50000, 100000]
STAGE12_LAMBDAS = [0.0005, 0.0025, 0.0075]
LAMBDAS = [0.0001, 0.0005, 0.0010]
ALL_LAMBDAS = [0.0001, 0.0005, 0.0010, 0.0025, 0.0075]
CONTEXTS = [1, 2, 3, 4, 8, 16, 32]
ARCHS = ["original", "shallow", "deep"]
METHOD_CONFIG = "shallow_c2_stab"
BASELINE_CONFIG = "image_gs_14bits"

LAM_COLOR = {
    0.0001: "#d62728",
    0.0005: "#1f77b4",
    0.0010: "#2ca02c",
    0.0025: "#ff7f0e",
    0.0075: "#9467bd",
}
LAM_MARKER = {0.0001: "*", 0.0005: "o", 0.0010: "D", 0.0025: "^", 0.0075: "v"}
LAM_LS = {0.0001: "-", 0.0005: "--", 0.0010: "-.", 0.0025: ":", 0.0075: ":"}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    sns.set_theme(style="whitegrid", rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })


def nlabel(n):
    return f"{n // 1000}k"


def save(fig, out, name):
    fname = f"{name}_fontes_grandes"
    fig.savefig(os.path.join(out, fname + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, fname + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {fname}.pdf / .png")


def filter_lambdas(df, lambdas):
    mask = np.zeros(len(df), dtype=bool)
    for l in lambdas:
        mask |= np.isclose(df['lambda_rdo'], l)
    return df[mask].copy()


def plot_stage0_stabilize():
    stab_csv = "resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv"
    if not os.path.isfile(stab_csv):
        return
    df_st = pd.read_csv(stab_csv)
    df_st['stabilize_grads_str'] = df_st['stabilize_grads'].astype(str).str.strip().str.lower()
    for col in ['bpp', 'psnr', 'lambda_rdo']:
        df_st[col] = df_st[col].astype(float)
    df_st = filter_lambdas(df_st, STAGE12_LAMBDAS)
    st_imgs = sorted(df_st['image_name'].unique())
    comp_st_imgs = [img for img in st_imgs if len(df_st[df_st['image_name'] == img]) == 18]
    df_stc = df_st[df_st['image_name'].isin(comp_st_imgs)]
    df_stm = df_stc.groupby(['arm_arch', 'stabilize_grads_str', 'lambda_rdo'], as_index=False)[['bpp', 'psnr']].mean()

    fig_arch, axes_arch = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    for ax, arch in zip(axes_arch, ['original', 'shallow', 'deep']):
        b_sub = df_stm[(df_stm['arm_arch'] == arch) & (df_stm['stabilize_grads_str'] == 'false')].sort_values('lambda_rdo', ascending=False)
        s_sub = df_stm[(df_stm['arm_arch'] == arch) & (df_stm['stabilize_grads_str'] == 'true')].sort_values('lambda_rdo', ascending=False)
        bd_st = bjontegaard_rate(
            b_sub.sort_values('psnr')['bpp'].values, b_sub.sort_values('psnr')['psnr'].values,
            s_sub.sort_values('psnr')['bpp'].values, s_sub.sort_values('psnr')['psnr'].values
        )
        ax.plot(b_sub['bpp'], b_sub['psnr'], linestyle='--', marker='s', color='#1f77b4', linewidth=2.4, markersize=8, label='Sem Estabilização (Âncora)')
        ax.plot(s_sub['bpp'], s_sub['psnr'], linestyle='-', marker='o', color='#ff7f0e', linewidth=2.6, markersize=8, label=f'Com Estabilização [BD: {bd_st:+.2f}%]')
        ax.set_title(f"Arquitetura: {arch.capitalize()}", fontsize=15.5, fontweight="bold")
        ax.set_xlabel("Taxa Total — bpp", fontsize=14)
        ax.tick_params(axis='both', labelsize=12.5)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=12.2, loc="lower right", frameon=True)
    axes_arch[0].set_ylabel("PSNR Médio (dB)", fontsize=14)
    fig_arch.suptitle("Impacto da Estabilização de Gradientes por Arquitetura (PSNR vs. bpp — 45 imagens)", fontsize=17, fontweight="bold", y=1.03)
    fig_arch.tight_layout()
    save(fig_arch, OUT_DIR, "rd_psnr_painel_arquiteturas_stabilize")


def plot_stage1_contexts(df):
    ctx_cfgs = {c: f"shallow_c{c}_stab" for c in CONTEXTS}
    sub = df[(df['num_gaussians'] == 100000) & (df['config'].isin(ctx_cfgs.values()))].copy()
    sub = filter_lambdas(sub, STAGE12_LAMBDAS)
    sub['arm_context'] = sub['arm_context'].astype(int)
    sub['cat'] = sub['image_name'].apply(category)
    imgs = sorted(sub['image_name'].unique())
    mean_df = sub.groupby(['arm_context', 'lambda_rdo'], as_index=False)[['bpp', 'bpp_img', 'bpp_arm', 'psnr']].mean()

    ctx_colors = {1: "#ff7f0e", 2: "#d62728", 3: "#e377c2", 4: "#1f77b4", 8: "#2ca02c", 16: "#333333", 32: "#9467bd"}
    ctx_markers = {1: "v", 2: "*", 3: "P", 4: "o", 8: "s", 16: "D", 32: "^"}
    ctx_ls = {1: "--", 2: "-", 3: "-.", 4: "--", 8: "-.", 16: "-", 32: ":"}

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), sharey=True)
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
                linewidth=3.0 if c == 2 else (2.4 if c == 16 else 2.0),
                markersize=11 if c == 2 else 7.5,
                label=lbl
            )
        ax.set_title(f"PSNR vs. bpp — {t_suf}", fontsize=15, fontweight="bold")
        ax.set_xlabel(x_lbl, fontsize=14)
        ax.tick_params(axis='both', labelsize=12.5)
        ax.grid(True, ls="--", alpha=0.6)
        ax.legend(fontsize=11.5, loc="lower right", frameon=True)
    axes[0].set_ylabel("PSNR Médio (dB)", fontsize=14)
    fig.suptitle(f"Etapa 1: Análise de Sensibilidade ao Contexto $C \\in \\{{1, 2, 3, 4, 8, 16, 32\\}}$ (Shallow, N=100k — {len(imgs)} imagens)", fontsize=16.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, OUT_DIR, "etapa1_contextos_total_vs_img")

    cats = sorted(sub['cat'].unique())
    fig_c, axes_c = plt.subplots(2, 3, figsize=(18, 11.2))
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
                linewidth=2.8 if c == 2 else 2.0, markersize=10 if c == 2 else 7,
                label=f"C={c} [{bd_txt}]"
            )
        ttl = f"Média Global ({len(imgs)} imagens)" if cat == '__global__' else f"Categoria: {CAT_PT.get(cat, cat)} (9 imagens)"
        ax.set_title(ttl, fontsize=15, fontweight="bold")
        ax.set_xlabel("Taxa Total — bpp", fontsize=13.5)
        ax.set_ylabel("PSNR (dB)", fontsize=13.5)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=11.0, loc="lower right", frameon=True)
        ax.grid(True, ls="--", alpha=0.6)
    fig_c.suptitle("Etapa 1: Curvas Taxa-Distorção por Categoria — Contextos $C \\in \\{1, 2, 3, 4, 8, 16, 32\\}$", fontsize=17.5, fontweight="bold", y=1.02)
    fig_c.tight_layout()
    save(fig_c, OUT_DIR, "etapa1_contextos_por_categoria")


def plot_stage2_archs(df):
    arch_cfgs = {a: f"{a}_c2_stab" for a in ARCHS}
    sub = df[(df['num_gaussians'] == 100000) & (df['config'].isin(arch_cfgs.values()))].copy()
    sub = filter_lambdas(sub, STAGE12_LAMBDAS)
    sub['cat'] = sub['image_name'].apply(category)
    imgs = sorted(sub['image_name'].unique())
    mean_df = sub.groupby(['arm_arch', 'lambda_rdo'], as_index=False)[['bpp', 'bpp_img', 'bpp_arm', 'psnr']].mean()

    arch_colors = {"original": "#1b7a5a", "shallow": "#d62728", "deep": "#5a4bb5"}
    arch_markers = {"original": "o", "shallow": "s", "deep": "^"}
    arch_ls = {"original": "--", "shallow": "-", "deep": "-."}

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)
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
                linewidth=2.8 if a == 'shallow' else 2.2, markersize=9.5 if a == 'shallow' else 7.5,
                label=f"{a.capitalize()} C=2 (arm={arm_v:.4f}) [{bd_txt}]"
            )
        ax.set_title(f"PSNR vs. bpp — {t_suf}", fontsize=15, fontweight="bold")
        ax.set_xlabel(x_lbl, fontsize=14)
        ax.tick_params(axis='both', labelsize=12.5)
        ax.grid(True, ls="--", alpha=0.6)
        ax.legend(fontsize=12.5, loc="lower right", frameon=True)
    axes[0].set_ylabel("PSNR Médio (dB)", fontsize=14)
    fig.suptitle(f"Etapa 2: Validação das Arquiteturas Fixando C=2 (N=100k — {len(imgs)} imagens)", fontsize=16.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, OUT_DIR, "etapa2_arquiteturas_c2_total_vs_img")


def plot_stage3_and_zoom(df, base, ours):
    imgs_b = set(complete_images(base, 'num_gaussians', N_LIST))
    imgs_o = set(complete_images(ours, 'num_gaussians', N_LIST))
    imgs_o = {i for i in imgs_o
              if all(len(ours[(ours.image_name == i) & (np.isclose(ours.lambda_rdo, l))]) == len(N_LIST) for l in ALL_LAMBDAS)}
    imgs = sorted(imgs_b & imgs_o)
    n_img = len(imgs)

    Rb, Pb = to_matrix(base, imgs, 'num_gaussians', N_LIST)
    mats = {}
    for lam in ALL_LAMBDAS:
        sub = ours[np.isclose(ours.lambda_rdo, lam)]
        mats[lam] = to_matrix(sub, imgs, 'num_gaussians', N_LIST)

    bd_glob = {}
    for lam in ALL_LAMBDAS:
        R, P = mats[lam]
        bd_glob[lam] = bootstrap_bd(Rb, Pb, R, P, n_boot=200)

    cats = sorted({category(i) for i in imgs})
    bd_cat = {}
    for cat in cats:
        rows = [k for k, i in enumerate(imgs) if category(i) == cat]
        bd_cat[cat] = {lam: bd_from_matrices(Rb, Pb, mats[lam][0], mats[lam][1], rows=rows) for lam in ALL_LAMBDAS}

    # 3A) Gráfico Global Suave (salvo como etapa3_rd_final_suave e etapa3_rd_final_global)
    for out_name in ("etapa3_rd_final_suave", "etapa3_rd_final_global"):
        fig, ax = plt.subplots(figsize=(9.2, 6.5))
        ax.plot(Rb.mean(0), Pb.mean(0), 'k-s', lw=2.6, ms=8.5, label="Image-GS original (baseline)")
        for j, n in enumerate(N_LIST):
            ax.annotate(f"N={nlabel(n)}", (Rb[:, j].mean(), Pb[:, j].mean()), textcoords="offset points", xytext=(6, -14), fontsize=11.5, fontweight="bold")
        for lam in LAMBDAS:
            R, P = mats[lam]
            b = bd_glob[lam][0]
            bd_lbl = f"BD: {b:+.2f}%" if not np.isnan(b) else "alta compressão"
            ax.plot(
                R.mean(0), P.mean(0), linestyle='-' if lam == 0.0001 else ('--' if lam == 0.0005 else '-.'),
                marker=LAM_MARKER[lam], color=LAM_COLOR[lam], lw=2.6 if lam == 0.0001 else 2.2,
                ms=10 if lam == 0.0001 else 8,
                label=f"GaussianARM (Shallow, C=2), $\\lambda$={lam:g}  [{bd_lbl}]"
            )
        ax.set_xlabel("Taxa total — bpp", fontsize=14); ax.set_ylabel("PSNR médio (dB)", fontsize=14)
        ax.set_title(f"Etapa 3: Taxa-distorção Final — Regime Suave ($\\lambda \\in \\{{0.0001, 0.0005, 0.0010\\}}$) — {n_img} imagens", fontweight="bold", fontsize=14.5)
        ax.tick_params(axis='both', labelsize=12.5)
        ax.grid(True, ls="--", alpha=0.6); ax.legend(fontsize=11.5, loc="lower right", frameon=True)
        save(fig, OUT_DIR, out_name)

    # 3B) Painel 2x3 por Categoria
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.2)); axes = axes.flatten()
    for k, cat in enumerate(cats + ['__global__']):
        ax = axes[k]
        rows = list(range(n_img)) if cat == '__global__' else [i for i, im in enumerate(imgs) if category(im) == cat]
        ax.plot(Rb[rows].mean(0), Pb[rows].mean(0), 'k-s', lw=2.4, ms=7.5, label="Image-GS (baseline)")
        for lam in LAMBDAS:
            R, P = mats[lam]
            b = bd_glob[lam][0] if cat == '__global__' else bd_cat[cat][lam]
            bd_lbl = f"BD: {b:+.2f}%" if not np.isnan(b) else "BD: N/A"
            ax.plot(
                R[rows].mean(0), P[rows].mean(0),
                linestyle='-' if lam == 0.0001 else ('--' if lam == 0.0005 else '-.'),
                marker=LAM_MARKER[lam], color=LAM_COLOR[lam], lw=2.4 if lam == 0.0001 else 2.0,
                ms=8.5 if lam == 0.0001 else 7, label=f"$\\lambda$={lam:g} [{bd_lbl}]"
            )
        title = f"Média global ({n_img} imagens)" if cat == '__global__' else f"Categoria: {CAT_PT.get(cat, cat)} ({len(rows)} imagens)"
        ax.set_title(title, fontweight="bold", fontsize=15); ax.set_xlabel("Taxa total — bpp", fontsize=13.5); ax.set_ylabel("PSNR (dB)", fontsize=13.5)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, ls="--", alpha=0.6); ax.legend(fontsize=12.0, loc="lower right", frameon=True)
    fig.suptitle("Etapa 3: Image-GS original vs. GaussianARM (Shallow, C=2) — Regime Suave por Categoria", fontsize=17.5, fontweight="bold", y=1.02)
    fig.tight_layout(); save(fig, OUT_DIR, "etapa3_rd_final_por_categoria")

    # 3C) Zoom por N (2x4) com fontes ampliadas
    rb_mean = Rb.mean(0)
    pb_mean = Pb.mean(0)
    interp_base = PchipInterpolator(np.log10(rb_mean), pb_mean)
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.6))

    for j, n in enumerate(N_LIST):
        ax_top = axes[0, j]
        r_soft = np.array([mats[l][0][:, j].mean() for l in LAMBDAS])
        p_soft = np.array([mats[l][1][:, j].mean() for l in LAMBDAS])

        r_min, r_max = r_soft.min(), r_soft.max()
        pad_r = max((r_max - r_min) * 0.35, 0.0015)
        x_grid = np.linspace(r_min - pad_r * 0.6, r_max + pad_r * 0.6, 100)
        y_base_grid = interp_base(np.log10(x_grid))
        y_base_at_soft = interp_base(np.log10(r_soft))

        ax_top.plot(x_grid, y_base_grid, 'k--', lw=2.4, label="Image-GS (mesma taxa)")
        ax_top.plot(r_soft, p_soft, '-', color="#444444", lw=2.0, alpha=0.7, zorder=3)
        ax_top.fill_between(
            r_soft, y_base_at_soft, p_soft,
            where=(p_soft >= y_base_at_soft),
            color="#2ca02c", alpha=0.14, interpolate=True, label="Ganho vs. Baseline"
        )

        for idx_l, lam in enumerate(LAMBDAS):
            rv, pv = r_soft[idx_l], p_soft[idx_l]
            dp = pv - y_base_at_soft[idx_l]
            ax_top.plot(
                rv, pv, marker=LAM_MARKER[lam], color=LAM_COLOR[lam],
                markersize=11 if lam == 0.0001 else 9.5, linestyle="None", zorder=5,
                label=f"$\\lambda={lam:g}$ ({dp:+.2f} dB)"
            )
            offsets = [(-22, 10), (-20, 10), (8, 8)]
            ax_top.annotate(
                f"$\\lambda={lam:g}$\n({dp:+.2f} dB)",
                (rv, pv), textcoords="offset points", xytext=offsets[idx_l],
                fontsize=10.8, fontweight="bold", color=LAM_COLOR[lam]
            )

        y_all_top = np.concatenate([p_soft, y_base_at_soft])
        y_span_top = max(y_all_top.max() - y_all_top.min(), 0.5)
        ax_top.set_xlim(r_min - pad_r * 0.8, r_max + pad_r * 0.9)
        ax_top.set_ylim(y_all_top.min() - y_span_top * 0.62, y_all_top.max() + y_span_top * 0.42)
        ax_top.set_title(
            f"$N = {nlabel(n)}$ — Regime Suave ($\\lambda \\leq 0.001$)\n"
            f"(Baseline: {rb_mean[j]:.3f} bpp / {pb_mean[j]:.2f} dB)",
            fontsize=13, fontweight="bold"
        )
        ax_top.set_xlabel("Taxa Total — bpp (Zoom)", fontsize=13)
        if j == 0:
            ax_top.set_ylabel("PSNR Médio (dB)", fontsize=13.5)
        ax_top.tick_params(axis='both', labelsize=11.5)
        ax_top.grid(True, ls="--", alpha=0.6)
        ax_top.legend(fontsize=10.5, loc="lower right", frameon=True)

        ax_bot = axes[1, j]
        r_all5 = np.array([mats[l][0][:, j].mean() for l in ALL_LAMBDAS])
        p_all5 = np.array([mats[l][1][:, j].mean() for l in ALL_LAMBDAS])

        r5_min, r5_max = r_all5.min(), r_all5.max()
        pad5_r = max((r5_max - r5_min) * 0.18, 0.003)
        x5_grid = np.linspace(r5_min - pad5_r * 0.5, r5_max + pad5_r * 0.5, 100)
        y5_base_grid = interp_base(np.log10(x5_grid))
        y5_base_at_pts = interp_base(np.log10(r_all5))

        ax_bot.plot(x5_grid, y5_base_grid, 'k--', lw=2.4, label="Image-GS (mesma taxa)")
        ax_bot.plot(r_all5, p_all5, '-', color="#555555", lw=2.0, alpha=0.75, zorder=3)

        for idx_l, lam in enumerate(ALL_LAMBDAS):
            rv, pv = r_all5[idx_l], p_all5[idx_l]
            dp = pv - y5_base_at_pts[idx_l]
            ax_bot.plot(
                rv, pv, marker=LAM_MARKER[lam], color=LAM_COLOR[lam],
                markersize=10.5 if lam == 0.0001 else 9, linestyle="None", zorder=5,
                label=f"$\\lambda={lam:g}$ ({dp:+.2f} dB)"
            )

        y_all_bot = np.concatenate([p_all5, y5_base_at_pts])
        y_span_bot = max(y_all_bot.max() - y_all_bot.min(), 1.0)
        ax_bot.set_xlim(r5_min - pad5_r * 0.6, r5_max + pad5_r * 0.6)
        ax_bot.set_ylim(y_all_bot.min() - y_span_bot * 0.15, y_all_bot.max() + y_span_bot * 0.18)
        ax_bot.set_title(f"$N = {nlabel(n)}$ — Todos os 5 $\\lambda$ ($0.0001$ a $0.0075$)", fontsize=13, fontweight="bold")
        ax_bot.set_xlabel("Taxa Total — bpp (Zoom)", fontsize=13)
        if j == 0:
            ax_bot.set_ylabel("PSNR Médio (dB)", fontsize=13.5)
        ax_bot.tick_params(axis='both', labelsize=11.5)
        ax_bot.grid(True, ls="--", alpha=0.6)
        ax_bot.legend(fontsize=10.2, loc="lower right", frameon=True)

    fig.suptitle(
        f"Etapa 3: Zoom por Número de Gaussianas $N$ — Efeito Local de $\\lambda$ vs. Curva Interpolada do Image-GS ({n_img} imagens)",
        fontsize=17.5, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    save(fig, OUT_DIR, "etapa3_zoom_por_N")


def plot_clic2025():
    csv_path = "resultados_diego_csvs/results_clic2025.csv"
    if not os.path.isfile(csv_path):
        return
    df = pd.read_csv(csv_path)
    for c in ['num_gaussians', 'lambda_rdo', 'psnr', 'bpp']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    base = df[df['config'] == BASELINE_CONFIG].copy()
    ours_all = df[df['config'] == METHOD_CONFIG].copy()
    avail_lambdas = [0.0001, 0.0005, 0.0075]
    imgs_b = set(complete_images(base, 'num_gaussians', N_LIST))
    imgs_per_lam = {l: set(complete_images(ours_all[np.isclose(ours_all['lambda_rdo'], l)], 'num_gaussians', N_LIST)) for l in avail_lambdas}
    imgs = sorted(imgs_b.intersection(*(imgs_per_lam[l] for l in avail_lambdas)))
    n_img = len(imgs)

    Rb, Pb = to_matrix(base, imgs, 'num_gaussians', N_LIST)
    mats = {l: to_matrix(ours_all[np.isclose(ours_all['lambda_rdo'], l)], imgs, 'num_gaussians', N_LIST) for l in avail_lambdas}
    bd_stats = {l: bootstrap_bd(Rb, Pb, mats[l][0], mats[l][1], n_boot=500) for l in avail_lambdas}

    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    ax.plot(Rb.mean(0), Pb.mean(0), 'k-s', lw=2.6, ms=8.5, label="Image-GS original (baseline, 14 bits)")
    for j, n in enumerate(N_LIST):
        ax.annotate(f"N={nlabel(n)}", (Rb[:, j].mean(), Pb[:, j].mean()), textcoords="offset points", xytext=(6, -14), fontsize=11.5, fontweight="bold")
    for lam in avail_lambdas:
        Ro, Po = mats[lam]
        pt = bd_stats[lam][0]
        bd_lbl = f"BD: {pt:+.2f}%" if not np.isnan(pt) else "alta compressão"
        ax.plot(
            Ro.mean(0), Po.mean(0), linestyle=LAM_LS.get(lam, '-'), marker=LAM_MARKER.get(lam, 'o'),
            color=LAM_COLOR.get(lam, '#333333'), lw=2.6 if lam == 0.0001 else 2.2, ms=10 if lam == 0.0001 else 8,
            label=f"GaussianARM (Shallow, C=2), $\\lambda={lam:g}$  [{bd_lbl}]"
        )
    ax.set_xlabel("Taxa Total — Bits por Pixel (bpp)", fontsize=14)
    ax.set_ylabel("PSNR Médio (dB)", fontsize=14)
    ax.set_title(f"CLIC 2025 ({n_img} imagens): Comparação dos {len(avail_lambdas)} Valores de $\\lambda$ vs. Image-GS", fontweight="bold", fontsize=14.5)
    ax.tick_params(axis='both', labelsize=12.5)
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend(fontsize=11.5, loc="lower right", frameon=True)
    save(fig, OUT_DIR, "rd_clic2025_todos_lambdas")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_style()
    df = pd.read_csv("resultados_diego_csvs/results_final_comparison.csv")
    for c in ['num_gaussians', 'lambda_rdo', 'psnr', 'bpp', 'bpp_img', 'bpp_arm']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    base = df[df['config'] == BASELINE_CONFIG].copy()
    ours = df[df['config'] == METHOD_CONFIG].copy()

    plot_stage0_stabilize()
    plot_stage1_contexts(df)
    plot_stage2_archs(df)
    plot_stage3_and_zoom(df, base, ours)
    plot_clic2025()


if __name__ == "__main__":
    main()
