"""Análise estatística (BD-Rate + Bootstrap IC 95%) e gráficos do benchmark CLIC 2025
para múltiplos valores de lambda (ex.: 0.0001 [melhor], 0.0005 [imediatamente inferior], 0.0075 [mais agressivo]).

Pode ser executado a qualquer momento (mesmo com o experimento ainda em andamento),
pois filtra automaticamente todas as imagens que já completaram os 4 valores de N
no Baseline (image_gs_14bits) e nos lambdas avaliados de (shallow_c2_stab).

Entrada : resultados_diego_csvs/results_clic2025.csv
Saída   : resultados_clic2025/rd_clic2025_global.{png,pdf}
          resultados_clic2025/rd_clic2025_todos_lambdas.{png,pdf}
          resultados_clic2025/zoom_por_N_clic2025.{png,pdf}
          resultados_clic2025/distribuicao_bdrate_clic2025.{png,pdf}
          resultados_clic2025/tabelas_clic2025.tex
          resultados_clic2025/resumo_clic2025.md

Uso:
    python scripts/calc_clic2025.py [--csv CAMINHO] [--out DIR] [--boot 2000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _analysis_utils import (fnum, pm, complete_images, to_matrix,  # noqa: E402
                             bootstrap_bd, psnr_overlap, per_image_bd)

N_LIST = [10000, 30000, 50000, 100000]
BASELINE_CONFIG = "image_gs_14bits"
WINNER_CONFIG = "shallow_c2_stab"
DEFAULT_LAMBDAS = [0.0001, 0.0005, 0.0075]

LAM_COLOR = {
    0.0001: "#d62728",  # Vermelho (Melhor resultado)
    0.0005: "#1f77b4",  # Azul (Imediatamente inferior)
    0.0010: "#2ca02c",  # Verde
    0.0025: "#ff7f0e",  # Laranja
    0.0075: "#9467bd",  # Roxo (Mais agressivo)
}
LAM_MARKER = {0.0001: "*", 0.0005: "o", 0.0010: "D", 0.0025: "s", 0.0075: "^"}
LAM_LS = {0.0001: "-", 0.0005: "--", 0.0010: "-.", 0.0025: ":", 0.0075: ":"}


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


def plot_zoom_by_n(Rb, Pb, mats, lambdas, out_dir, n_img):
    """Painel 2x4 com zoom na faixa de BPP de cada N in {10k, 30k, 50k, 100k}:
       - Linha 1: Zoom nos lambdas suaves (<= 0.001) vs. curva PCHIP do Baseline na mesma taxa.
       - Linha 2: Todos os lambdas avaliados (incluindo 0.0075 agressivo).
    """
    rb_mean = Rb.mean(0)
    pb_mean = Pb.mean(0)
    interp_base = PchipInterpolator(np.log10(rb_mean), pb_mean)
    soft_lams = [l for l in lambdas if l <= 0.001] or lambdas

    fig, axes = plt.subplots(2, 4, figsize=(20, 9.6))

    for j, n in enumerate(N_LIST):
        # Linha 0: Regime suave
        ax_top = axes[0, j]
        r_soft = np.array([mats[l][0][:, j].mean() for l in soft_lams])
        p_soft = np.array([mats[l][1][:, j].mean() for l in soft_lams])

        r_min, r_max = r_soft.min(), r_soft.max()
        pad_r = max((r_max - r_min) * 0.45, 0.002)
        x_grid = np.linspace(r_min - pad_r * 0.6, r_max + pad_r * 0.6, 100)
        y_base_grid = interp_base(np.log10(x_grid))
        y_base_at_soft = interp_base(np.log10(r_soft))

        ax_top.plot(x_grid, y_base_grid, 'k--', lw=2.2, label="Image-GS (curva na mesma taxa)")
        ax_top.plot(r_soft, p_soft, '-', color="#444444", lw=1.8, alpha=0.7, zorder=3)
        ax_top.fill_between(
            r_soft, y_base_at_soft, p_soft,
            where=(p_soft >= y_base_at_soft),
            color="#2ca02c", alpha=0.14, interpolate=True, label="Ganho sobre Baseline"
        )

        for idx_l, lam in enumerate(soft_lams):
            rv, pv = r_soft[idx_l], p_soft[idx_l]
            dp = pv - y_base_at_soft[idx_l]
            col = LAM_COLOR.get(lam, "#333333")
            mk = LAM_MARKER.get(lam, "o")
            ax_top.plot(
                rv, pv, marker=mk, color=col,
                markersize=10 if lam == 0.0001 else 8.5, linestyle="None", zorder=5,
                label=f"$\\lambda={lam:g}$ ({dp:+.2f} dB)"
            )
            off = (-18, 9) if idx_l == 0 else (8, 8)
            ax_top.annotate(
                f"$\\lambda={lam:g}$\n({dp:+.2f} dB)",
                (rv, pv), textcoords="offset points", xytext=off,
                fontsize=8.2, fontweight="bold", color=col
            )

        y_all_top = np.concatenate([p_soft, y_base_at_soft])
        y_span_top = max(y_all_top.max() - y_all_top.min(), 0.5)
        ax_top.set_xlim(r_min - pad_r * 0.75, r_max + pad_r * 0.85)
        ax_top.set_ylim(y_all_top.min() - y_span_top * 0.48, y_all_top.max() + y_span_top * 0.35)
        ax_top.set_title(
            f"$N = {nlabel(n)}$ — Regime Suave\n(Base: {rb_mean[j]:.3f} bpp / {pb_mean[j]:.2f} dB)",
            fontsize=11, fontweight="bold"
        )
        ax_top.set_xlabel("Taxa Total — bpp (Zoom)", fontsize=10)
        if j == 0:
            ax_top.set_ylabel("PSNR Médio (dB)", fontsize=10.5)
        ax_top.grid(True, ls="--", alpha=0.6)
        ax_top.legend(fontsize=7.8, loc="lower right", frameon=True)

        # Linha 1: Todos os lambdas (incluindo 0.0075 agressivo)
        ax_bot = axes[1, j]
        r_all = np.array([mats[l][0][:, j].mean() for l in lambdas])
        p_all = np.array([mats[l][1][:, j].mean() for l in lambdas])

        r_min_b, r_max_b = r_all.min(), r_all.max()
        pad_b = max((r_max_b - r_min_b) * 0.18, 0.003)
        x_grid_b = np.linspace(r_min_b - pad_b * 0.5, r_max_b + pad_b * 0.5, 100)
        y_base_grid_b = interp_base(np.log10(x_grid_b))
        y_base_at_all = interp_base(np.log10(r_all))

        ax_bot.plot(x_grid_b, y_base_grid_b, 'k--', lw=2.2, label="Image-GS (curva na mesma taxa)")
        ax_bot.plot(r_all, p_all, '-', color="#555555", lw=1.8, alpha=0.75, zorder=3)

        for idx_l, lam in enumerate(lambdas):
            rv, pv = r_all[idx_l], p_all[idx_l]
            dp = pv - y_base_at_all[idx_l]
            col = LAM_COLOR.get(lam, "#333333")
            mk = LAM_MARKER.get(lam, "o")
            ax_bot.plot(
                rv, pv, marker=mk, color=col,
                markersize=9.5 if lam == 0.0001 else 8, linestyle="None", zorder=5,
                label=f"$\\lambda={lam:g}$ ({dp:+.2f} dB)"
            )

        y_all_bot = np.concatenate([p_all, y_base_at_all])
        y_span_bot = max(y_all_bot.max() - y_all_bot.min(), 1.0)
        ax_bot.set_xlim(r_min_b - pad_b * 0.6, r_max_b + pad_b * 0.6)
        ax_bot.set_ylim(y_all_bot.min() - y_span_bot * 0.12, y_all_bot.max() + y_span_bot * 0.15)
        ax_bot.set_title(f"$N = {nlabel(n)}$ — Todos os $\\lambda$ ({lambdas[0]:g} a {lambdas[-1]:g})", fontsize=11, fontweight="bold")
        ax_bot.set_xlabel("Taxa Total — bpp (Zoom)", fontsize=10)
        if j == 0:
            ax_bot.set_ylabel("PSNR Médio (dB)", fontsize=10.5)
        ax_bot.grid(True, ls="--", alpha=0.6)
        ax_bot.legend(fontsize=7.8, loc="lower right", frameon=True)

    fig.suptitle(
        f"CLIC 2025: Zoom por Número de Gaussianas $N$ — Efeito Local de $\\lambda$ vs. Curva Interpolada do Image-GS ({n_img} imagens)",
        fontsize=14.5, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    save(fig, out_dir, "zoom_por_N_clic2025")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default="resultados_diego_csvs/results_clic2025.csv")
    ap.add_argument('--out', default="resultados_clic2025")
    ap.add_argument('--lambdas', nargs='*', type=float, default=None)
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV ainda não encontrado em '{args.csv}'. Inicie `python scripts/run_clic2025.py` primeiro.")
        return

    os.makedirs(args.out, exist_ok=True)
    setup_style()

    df = pd.read_csv(args.csv)
    for c in ['num_gaussians', 'lambda_rdo', 'psnr', 'bpp', 'bpp_img', 'bpp_arm', 'msssim', 'lpips', 'time_s']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    base = df[df['config'] == BASELINE_CONFIG].copy()
    ours_all = df[df['config'] == WINNER_CONFIG].copy()

    lambdas = sorted(args.lambdas or DEFAULT_LAMBDAS)
    imgs_b = set(complete_images(base, 'num_gaussians', N_LIST))

    # Filtra imagens que já têm os 4 Ns completos no baseline e em cada lambda disponível
    avail_lambdas = []
    imgs_per_lam = {}
    for lam in lambdas:
        sub_l = ours_all[np.isclose(ours_all['lambda_rdo'], lam)]
        c_imgs = set(complete_images(sub_l, 'num_gaussians', N_LIST))
        if c_imgs:
            avail_lambdas.append(lam)
            imgs_per_lam[lam] = c_imgs

    if not avail_lambdas or not imgs_b:
        print("Ainda não há imagens com os 4 valores de N concluídos no Baseline e no Método.")
        return

    imgs = sorted(imgs_b.intersection(*(imgs_per_lam[l] for l in avail_lambdas)))
    n_img = len(imgs)
    print(f"Imagens completas no Baseline: {len(imgs_b)} | Completas em todos os lambdas {avail_lambdas}: {n_img}")
    if n_img == 0:
        print("Nenhuma imagem completou todos os lambdas ainda.")
        return

    Rb, Pb = to_matrix(base, imgs, 'num_gaussians', N_LIST)
    mats = {}
    for lam in avail_lambdas:
        sub = ours_all[np.isclose(ours_all['lambda_rdo'], lam)]
        mats[lam] = to_matrix(sub, imgs, 'num_gaussians', N_LIST)

    # 1. BD-Rate Global + IC 95% via Bootstrap para cada lambda
    print("\n=== Resultado Global CLIC 2025 por Lambda ===")
    bd_stats = {}
    for lam in avail_lambdas:
        Ro, Po = mats[lam]
        pt, lo, hi = bootstrap_bd(Rb, Pb, Ro, Po, n_boot=args.boot)
        a, b, w = psnr_overlap(Pb, Po)
        v_img = per_image_bd(Rb, Pb, Ro, Po)
        ok = v_img[~np.isnan(v_img)]
        wins = int(np.sum(ok < 0))
        n_valid = len(ok)
        med_bd = float(np.median(ok)) if n_valid > 0 else float('nan')
        q25, q75 = (np.percentile(ok, [25, 75]) if n_valid > 0 else (float('nan'), float('nan')))
        bd_stats[lam] = dict(pt=pt, lo=lo, hi=hi, a=a, b=b, w=w, v_img=v_img, ok=ok,
                             wins=wins, n_valid=n_valid, med=med_bd, q25=q25, q75=q75)
        print(f"  * lambda={lam:<7g}: BD-Rate Global = {pt:+.2f}% (IC95% [{lo:+.2f}%; {hi:+.2f}%]) | "
              f"Mediana por img = {med_bd:+.2f}% | Vitórias: {wins}/{n_valid} ({100*wins/max(n_valid,1):.1f}%) | "
              f"Sobreposição PSNR: {a:.1f}-{b:.1f} dB ({w:.1f} dB)")

    # 2. Comparação pareada por N e lambda
    print("\n=== Comparação Pareada por N e Lambda ===")
    pair_rows = []
    for j, n in enumerate(N_LIST):
        b_sub = base[(base['num_gaussians'] == n) & (base['image_name'].isin(imgs))]
        bb, bp = Rb[:, j].mean(), Pb[:, j].mean()
        b_ms, b_lp = b_sub['msssim'].mean(), b_sub['lpips'].mean()
        for lam in avail_lambdas:
            Ro, Po = mats[lam]
            o_sub = ours_all[(ours_all['num_gaussians'] == n) & np.isclose(ours_all['lambda_rdo'], lam) & (ours_all['image_name'].isin(imgs))]
            ob, op = Ro[:, j].mean(), Po[:, j].mean()
            dbpp = (ob / bb - 1.0) * 100.0
            dpsnr = op - bp
            pair_rows.append((
                n, lam, bb, bp, b_ms, b_lp,
                ob, op, o_sub['msssim'].mean(), o_sub['lpips'].mean(),
                o_sub['bpp_img'].mean(), o_sub['bpp_arm'].mean(), dbpp, dpsnr
            ))
            print(f"  N={nlabel(n):>4s} L={lam:<7g}: Base {bb:.4f} bpp / {bp:.2f} dB | Nosso {ob:.4f} bpp / {op:.2f} dB | Δbpp {dbpp:+.2f}% | ΔPSNR {dpsnr:+.2f} dB")

    # 3. Gráfico 1: Curva RD Global (Lambdas Suaves <= 0.001)
    soft_lams = [l for l in avail_lambdas if l <= 0.001] or avail_lambdas
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.plot(Rb.mean(0), Pb.mean(0), 'k-s', lw=2.4, ms=8, label="Image-GS original (baseline, 14 bits)")
    for j, n in enumerate(N_LIST):
        ax.annotate(f"N={nlabel(n)}", (Rb[:, j].mean(), Pb[:, j].mean()), textcoords="offset points", xytext=(6, -12), fontsize=8.8)
    for lam in soft_lams:
        Ro, Po = mats[lam]
        st = bd_stats[lam]
        bd_lbl = f"BD: {st['pt']:+.2f}%, IC95% [{st['lo']:+.2f}%; {st['hi']:+.2f}%]" if not np.isnan(st['pt']) else "alta compressão"
        ax.plot(
            Ro.mean(0), Po.mean(0), linestyle=LAM_LS.get(lam, '-'), marker=LAM_MARKER.get(lam, 'o'),
            color=LAM_COLOR.get(lam, '#d62728'), lw=2.5 if lam == 0.0001 else 2.1, ms=10 if lam == 0.0001 else 7.5,
            label=f"GaussianARM (Shallow, C=2), $\\lambda={lam:g}$  [{bd_lbl}]"
        )
    ax.set_xlabel("Taxa Total — Bits por Pixel (bpp)", fontsize=11)
    ax.set_ylabel("PSNR Médio (dB)", fontsize=11)
    ax.set_title(f"CLIC 2025 ({n_img} imagens): Image-GS vs. GaussianARM (Shallow, C=2)", fontweight="bold", fontsize=12.5)
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend(fontsize=9.0, loc="lower right", frameon=True)
    save(fig, args.out, "rd_clic2025_global")

    # 4. Gráfico 2: Curva RD com TODOS os lambdas (incluindo o mais agressivo 0.0075)
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.plot(Rb.mean(0), Pb.mean(0), 'k-s', lw=2.5, ms=8, label="Image-GS original (baseline, 14 bits)")
    for j, n in enumerate(N_LIST):
        ax.annotate(f"N={nlabel(n)}", (Rb[:, j].mean(), Pb[:, j].mean()), textcoords="offset points", xytext=(6, -12), fontsize=8.8)
    for lam in avail_lambdas:
        Ro, Po = mats[lam]
        st = bd_stats[lam]
        bd_lbl = f"BD: {st['pt']:+.2f}%" if not np.isnan(st['pt']) else "alta compressão"
        ax.plot(
            Ro.mean(0), Po.mean(0), linestyle=LAM_LS.get(lam, '-'), marker=LAM_MARKER.get(lam, 'o'),
            color=LAM_COLOR.get(lam, '#333333'), lw=2.5 if lam == 0.0001 else 2.0, ms=9.5 if lam == 0.0001 else 7.5,
            label=f"GaussianARM (Shallow, C=2), $\\lambda={lam:g}$  [{bd_lbl}]"
        )
    ax.set_xlabel("Taxa Total — Bits por Pixel (bpp)", fontsize=11)
    ax.set_ylabel("PSNR Médio (dB)", fontsize=11)
    ax.set_title(f"CLIC 2025 ({n_img} imagens): Comparação dos {len(avail_lambdas)} Valores de $\\lambda$ vs. Image-GS", fontweight="bold", fontsize=12.5)
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend(fontsize=8.8, loc="lower right", frameon=True)
    save(fig, args.out, "rd_clic2025_todos_lambdas")

    # 5. Gráfico 3: Zoom por N
    plot_zoom_by_n(Rb, Pb, mats, avail_lambdas, args.out, n_img)

    # 6. Gráfico 4: Distribuição de BD-Rate por imagem para os lambdas com sobreposição válida
    valid_lams = [l for l in avail_lambdas if bd_stats[l]['n_valid'] > 1]
    if valid_lams:
        fig, axes = plt.subplots(1, len(valid_lams), figsize=(6.2 * len(valid_lams), 5.0), sharey=True)
        if len(valid_lams) == 1:
            axes = [axes]
        for ax, lam in zip(axes, valid_lams):
            st = bd_stats[lam]
            clipped = np.clip(st['ok'], -45, 45)
            sns.histplot(clipped, bins=15, kde=True, color=LAM_COLOR.get(lam, "#1f77b4"), ax=ax)
            ax.axvline(0.0, color="black", linestyle="--", lw=1.6, label="Empate (0%)")
            ax.axvline(st['med'], color="#d62728", linestyle="-", lw=2.0,
                       label=f"Mediana: {st['med']:+.2f}% ({st['wins']}/{st['n_valid']} < 0%)")
            ax.set_title(f"$\\lambda = {lam:g}$ (BD Global: {st['pt']:+.2f}%)", fontweight="bold", fontsize=11.5)
            ax.set_xlabel("BD-Rate por Imagem (%)", fontsize=10)
            ax.legend(fontsize=8.5, loc="upper right", frameon=True)
            ax.grid(True, ls="--", alpha=0.6)
        axes[0].set_ylabel("Número de Imagens", fontsize=10.5)
        fig.suptitle(f"Distribuição de BD-Rate por Imagem no CLIC 2025 ({n_img} imagens)", fontweight="bold", fontsize=13.5, y=1.02)
        fig.tight_layout()
        save(fig, args.out, "distribuicao_bdrate_clic2025")

    # 7. Tabelas LaTeX e Resumo Markdown
    tex = [
        "% ===== Tabela 1 CLIC 2025: BD-Rate por Lambda com IC 95% (Bootstrap) =====",
        "\\begin{table}[htbp]\n    \\centering",
        f"    \\caption{{Avaliação no conjunto de teste CLIC 2025 ({n_img} imagens): BD-Rate (PSNR) do \\textit{{GaussianARM}} (\\textit{{Shallow}}, $C=2$) em relação ao \\textit{{Image-GS}} original para os três regimes de $\\lambda$. Valores negativos indicam redução de taxa para a mesma qualidade visual.}}",
        "    \\label{tab:clic2025_bdrate}\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{|c|c|c|c|c|}\n        \\hline",
        "        $\\boldsymbol{\\lambda}$ & \\textbf{BD-Rate Global} & \\textbf{IC 95\\% (\\textit{Bootstrap})} & \\textbf{Mediana por Imagem} & \\textbf{Imagens com BD-Rate $< 0\\%$} \\\\ \\hline\\hline",
    ]
    for lam in avail_lambdas:
        st = bd_stats[lam]
        bd_s = pm(st['pt']) if not np.isnan(st['pt']) else "N/A (sem sobreposição)"
        ic_s = f"$[{fnum(st['lo'], 2, True)}\\%;\\, {fnum(st['hi'], 2, True)}\\%]$" if not np.isnan(st['lo']) else "---"
        med_s = pm(st['med']) if not np.isnan(st['med']) else "---"
        win_s = f"{st['wins']}/{st['n_valid']} ({100*st['wins']/max(st['n_valid'],1):.1f}\\%)" if st['n_valid'] > 0 else "---"
        tex.append(f"        {fnum(lam, 4)} & {bd_s} & {ic_s} & {med_s} & {win_s} \\\\ \\hline")
    tex.append("    \\end{tabular}\n    }\n    \\fonte O autor (2026).\n\\end{table}\n")

    tex.extend([
        "% ===== Tabela 2 CLIC 2025: Comparação Pareada por N e Lambda =====",
        "\\begin{table}[htbp]\n    \\centering",
        f"    \\caption{{Comparação pareada no conjunto CLIC 2025 ({n_img} imagens) por número de gaussianas $N$ e multiplicador $\\lambda$ entre o \\textit{{Image-GS}} original e o \\textit{{GaussianARM}} (\\textit{{Shallow}}, $C=2$).}}",
        "    \\label{tab:clic2025_pareada}\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{|c|c|ccc|ccc|c|c|}\n        \\hline",
        "        \\multirow{2}{*}{$\\boldsymbol{N}$} & \\multirow{2}{*}{$\\boldsymbol{\\lambda}$} & \\multicolumn{3}{c|}{\\textbf{\\textit{Image-GS} (Baseline)}} & \\multicolumn{3}{c|}{\\textbf{\\textit{GaussianARM} (\\textit{Shallow}, $C=2$)}} & \\multirow{2}{*}{$\\boldsymbol{\\Delta}$\\textbf{bpp}} & \\multirow{2}{*}{$\\boldsymbol{\\Delta}$\\textbf{PSNR (dB)}} \\\\ \\cline{3-8}",
        "        & & \\textbf{bpp ($\\downarrow$)} & \\textbf{PSNR (dB) ($\\uparrow$)} & \\textbf{MS-SSIM ($\\uparrow$)} & \\textbf{bpp ($\\downarrow$)} & \\textbf{PSNR (dB) ($\\uparrow$)} & \\textbf{MS-SSIM ($\\uparrow$)} & & \\\\ \\hline\\hline",
    ])
    for n in N_LIST:
        rows_n = [r for r in pair_rows if r[0] == n]
        for k, r in enumerate(rows_n):
            first = f"\\multirow{{{len(rows_n)}}}{{*}}{{{n:,}".replace(',', '.') + "}}" if k == 0 else ""
            sep = "\\cline{2-10}" if k < len(rows_n) - 1 else "\\hline"
            tex.append(f"        {first} & {fnum(r[1], 4)} & {fnum(r[2], 4)} & {fnum(r[3], 2)} & {fnum(r[4], 4)} & "
                       f"{fnum(r[6], 4)} & {fnum(r[7], 2)} & {fnum(r[8], 4)} & {pm(r[12], 1)} & {fnum(r[13], 2, True)} \\\\ {sep}")
    tex.append("    \\end{tabular}\n    }\n    \\fonte O autor (2026).\n\\end{table}\n")

    with open(os.path.join(args.out, "tabelas_clic2025.tex"), "w") as f:
        f.write("\n".join(tex))

    md = [
        f"# Resultados CLIC 2025 (`{WINNER_CONFIG}` — {n_img} imagens)\n",
        "## 1. BD-Rate Global e Bootstrap (IC 95%) por $\\lambda$\n",
        "| $\\lambda$ | BD-Rate Global | IC 95% (Bootstrap) | Mediana por Imagem | Vitórias (`BD < 0%`) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    for lam in avail_lambdas:
        st = bd_stats[lam]
        bd_s = f"**{st['pt']:+.2f}%**" if not np.isnan(st['pt']) else "N/A (sem sobreposição)"
        ic_s = f"`[{st['lo']:+.2f}%, {st['hi']:+.2f}%]`" if not np.isnan(st['lo']) else "---"
        med_s = f"**{st['med']:+.2f}%**" if not np.isnan(st['med']) else "---"
        win_s = f"**{st['wins']}/{st['n_valid']} ({100*st['wins']/max(st['n_valid'],1):.1f}%)**" if st['n_valid'] > 0 else "---"
        md.append(f"| **{lam:g}** | {bd_s} | {ic_s} | {med_s} | {win_s} |")

    md.extend([
        "\n## 2. Comparação Pareada por $N$ e $\\lambda$\n",
        "| $N$ | $\\lambda$ | Baseline BPP | Baseline PSNR (dB) | Baseline MS-SSIM | Nosso BPP (`bpp_img` + `bpp_arm`) | Nosso PSNR (dB) | Nosso MS-SSIM | $\\Delta$ BPP (%) | $\\Delta$ PSNR (dB) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in pair_rows:
        md.append(f"| **{r[0]}** | `{r[1]:g}` | {r[2]:.4f} | {r[3]:.2f} | {r[4]:.4f} | {r[6]:.4f} ({r[10]:.4f} + {r[11]:.4f}) | {r[7]:.2f} | {r[8]:.4f} | **{r[12]:+.2f}%** | {r[13]:+.2f} |")

    with open(os.path.join(args.out, "resumo_clic2025.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nArquivos salvos em '{args.out}/'!")


if __name__ == "__main__":
    main()
