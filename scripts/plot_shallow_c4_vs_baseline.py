import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    base_dir = "/home/diego/image-gs"
    out_dir = os.path.join(base_dir, "plots_arm_context")
    fig_dir = os.path.join(base_dir, "tcc_diego/figuras")
    art_dir = "/home/diego/.gemini/antigravity/brain/59567a6c-0bcf-496f-ba44-271dbb78ea6a"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    })
    sns.set_theme(style="whitegrid", rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
    })

    df_base = pd.read_csv(os.path.join(base_dir, "resultados_diego_csvs/resultados_experimentos.csv"))
    df_ctx = pd.read_csv(os.path.join(base_dir, "resultados_diego_csvs/results_shallow_stab_context_4_8_16_32.csv"))
    df_arch = pd.read_csv(os.path.join(base_dir, "resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv"))

    for col in ['num_gaussians', 'lambda_rdo', 'psnr', 'bpp', 'bpp_img', 'bpp_arm']:
        if col in df_base.columns:
            df_base[col] = df_base[col].astype(float)
        if col in df_ctx.columns:
            df_ctx[col] = df_ctx[col].astype(float)
        if col in df_arch.columns:
            df_arch[col] = df_arch[col].astype(float)

    imgs_5 = ['anime-2_2k', 'art-1_2k', 'painting-1_2k', 'photo-5_2k', 'vector-1_2k']
    cat_labels = {
        'anime-2_2k': 'Anime (anime-2_2k)',
        'art-1_2k': 'Arte (art-1_2k)',
        'painting-1_2k': 'Pintura (painting-1_2k)',
        'photo-5_2k': 'Fotografia (photo-5_2k)',
        'vector-1_2k': 'Vetorial (vector-1_2k)',
    }

    # Filtra dados das 5 imagens no baseline (lambda=0) e Fase 1 ARM (lambda=0.0005)
    base_curve_5 = df_base[np.isclose(df_base['lambda_rdo'], 0.0)].copy()
    p1_curve_5 = df_base[np.isclose(df_base['lambda_rdo'], 0.0005)].copy()
    c4_all = df_ctx[df_ctx['arm_context'] == 4].copy()
    c4_5 = c4_all[c4_all['image_name'].isin(imgs_5)].copy()

    # =========================================================================
    # GRÁFICO 1: Painel 2x3 (5 imagens representativas + Média das 5 imagens)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    axes = axes.flatten()

    for idx, img in enumerate(imgs_5):
        ax = axes[idx]
        b_sub = base_curve_5[base_curve_5['image_name'] == img].sort_values('num_gaussians')
        p1_sub = p1_curve_5[p1_curve_5['image_name'] == img].sort_values('num_gaussians')
        c4_sub = c4_5[c4_5['image_name'] == img].sort_values('lambda_rdo', ascending=False)

        # Curva Image-GS Original (10k..100k)
        ax.plot(
            b_sub['bpp'], b_sub['psnr'],
            linestyle='--', marker='s', color='#333333', linewidth=2.2, markersize=7,
            label='Image-GS Original (10k–100k)'
        )
        # Curva Fase 1 GaussianARM (Original C=16, s/ estab., 10k..100k, lambda=0.0005)
        ax.plot(
            p1_sub['bpp'], p1_sub['psnr'],
            linestyle=':', marker='^', color='#1f77b4', linewidth=2.0, markersize=7,
            label='GaussianARM Inicial (C=16, λ=0,0005)'
        )
        # Curva Fase 2 Shallow C=4 Estabilizado (N=100k, 3 lambdas)
        ax.plot(
            c4_sub['bpp'], c4_sub['psnr'],
            linestyle='-', marker='o', color='#d62728', linewidth=2.4, markersize=7.5,
            label='Shallow C=4 Estab. (100k, 3 λ)'
        )
        # Destaca o ponto ótimo N=100k, lambda=0.0005 do Shallow C=4
        pt_opt = c4_sub[np.isclose(c4_sub['lambda_rdo'], 0.0005)].iloc[0]
        pt_base100 = b_sub[b_sub['num_gaussians'] == 100000].iloc[0]
        sav_pct = (pt_opt['bpp'] / pt_base100['bpp'] - 1.0) * 100.0
        ax.scatter([pt_opt['bpp']], [pt_opt['psnr']], color='#d62728', s=110, zorder=6, edgecolors='black', linewidth=1.2)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + 0.08 * (y_max - y_min))
        ax.annotate(
            f"100k (λ=0,0005): {sav_pct:+.1f}% bpp".replace('.', ','),
            (pt_opt['bpp'], pt_opt['psnr']),
            textcoords="offset points", xytext=(-95, 10),
            fontsize=8.5, fontweight='bold', color='#b01515',
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff8f8", ec="#d62728", alpha=0.92)
        )

        ax.set_title(f"Categoria: {cat_labels[img]}", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Taxa Total — bpp", fontsize=10.5)
        ax.set_ylabel("PSNR (dB)", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="lower right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    # 6º subplot: Média das 5 imagens
    ax_m = axes[5]
    b_mean = base_curve_5.groupby('num_gaussians', as_index=False)[['bpp', 'psnr']].mean().sort_values('num_gaussians')
    p1_mean = p1_curve_5.groupby('num_gaussians', as_index=False)[['bpp', 'psnr']].mean().sort_values('num_gaussians')
    c4_mean5 = c4_5.groupby('lambda_rdo', as_index=False)[['bpp', 'psnr']].mean().sort_values('lambda_rdo', ascending=False)

    ax_m.plot(
        b_mean['bpp'], b_mean['psnr'],
        linestyle='--', marker='s', color='#333333', linewidth=2.4, markersize=7.5,
        label='Image-GS Original (10k–100k)'
    )
    ax_m.plot(
        p1_mean['bpp'], p1_mean['psnr'],
        linestyle=':', marker='^', color='#1f77b4', linewidth=2.2, markersize=7.5,
        label='GaussianARM Inicial (C=16, λ=0,0005)'
    )
    ax_m.plot(
        c4_mean5['bpp'], c4_mean5['psnr'],
        linestyle='-', marker='o', color='#d62728', linewidth=2.6, markersize=8,
        label='Shallow C=4 Estab. (100k, 3 λ)'
    )
    pt_opt_m = c4_mean5[np.isclose(c4_mean5['lambda_rdo'], 0.0005)].iloc[0]
    ax_m.scatter([pt_opt_m['bpp']], [pt_opt_m['psnr']], color='#d62728', s=120, zorder=6, edgecolors='black', linewidth=1.2)
    y_min_m, y_max_m = ax_m.get_ylim()
    ax_m.set_ylim(y_min_m, y_max_m + 0.08 * (y_max_m - y_min_m))
    ax_m.annotate(
        "100k (λ=0,0005): -28,6% bpp | +0,44 dB iso-taxa",
        (pt_opt_m['bpp'], pt_opt_m['psnr']),
        textcoords="offset points", xytext=(-135, 10),
        fontsize=8.5, fontweight='bold', color='#b01515',
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.2", fc="#fff8f8", ec="#d62728", alpha=0.92)
    )
    ax_m.set_title("Média das 5 Imagens de Referência", fontsize=12.5, fontweight="bold")
    ax_m.set_xlabel("Taxa Total — bpp", fontsize=10.5)
    ax_m.set_ylabel("PSNR (dB)", fontsize=10.5)
    ax_m.legend(fontsize=8.5, loc="lower right", frameon=True)
    ax_m.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Comparação Taxa-Distorção (PSNR vs. bpp) — Image-GS Original vs. GaussianARM Shallow (C=4)", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    for d in (out_dir, fig_dir):
        fig.savefig(os.path.join(d, "rd_psnr_shallow_c4_vs_baseline_painel.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, "rd_psnr_shallow_c4_vs_baseline_painel.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(art_dir, "rd_psnr_shallow_c4_vs_baseline_painel.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # GRÁFICO 2: Zoom em N=100k (Painel 1x2: Evolução pareada 5 imgs + 45 imgs)
    # =========================================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0))

    # Subplot (a): Curva completa 10k-100k com destaque para o salto em N=100k nas 5 imagens
    ax1.plot(
        b_mean['bpp'], b_mean['psnr'],
        linestyle='--', marker='s', color='#333333', linewidth=2.4, markersize=8,
        label='Image-GS Original (λ=0, 10k–100k)'
    )
    for _, r in b_mean.iterrows():
        n_k = int(r['num_gaussians'] // 1000)
        ax1.annotate(f"{n_k}k", (r['bpp'], r['psnr']), textcoords="offset points", xytext=(5, -11), fontsize=8.5, color='#333333')

    ax1.plot(
        p1_mean['bpp'], p1_mean['psnr'],
        linestyle=':', marker='^', color='#1f77b4', linewidth=2.2, markersize=7.5,
        label='GaussianARM Inicial (Original C=16, λ=0,0005)'
    )
    ax1.plot(
        c4_mean5['bpp'], c4_mean5['psnr'],
        linestyle='-', marker='o', color='#d62728', linewidth=2.5, markersize=8,
        label='GaussianARM Final (Shallow C=4 Estab., N=100k)'
    )
    for _, r in c4_mean5.iterrows():
        l_str = f"λ={r['lambda_rdo']:g}".replace('.', ',')
        off = (-38, 6) if np.isclose(r['lambda_rdo'], 0.0005) else (8, -4)
        ax1.annotate(l_str, (r['bpp'], r['psnr']), textcoords="offset points", xytext=off, fontsize=8.5, color='#d62728', fontweight='bold')

    # Seta indicando a redução de bpp em N=100k (de 2.6703 para 1.9074 bpp)
    b100 = b_mean[b_mean['num_gaussians'] == 100000].iloc[0]
    ax1.annotate(
        "", xy=(pt_opt_m['bpp'], pt_opt_m['psnr']), xytext=(b100['bpp'], b100['psnr']),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.0, ls="-.")
    )
    ax1.text(
        (b100['bpp'] + pt_opt_m['bpp']) / 2, (b100['psnr'] + pt_opt_m['psnr']) / 2 + 0.35,
        "-28,6% bpp (N=100k)", fontsize=9.5, fontweight='bold', color='#1b781b', ha='center'
    )

    ax1.set_title("(a) Comparação Pareada nas 5 Imagens (10k a 100k Gaussianas)", fontsize=12.5, fontweight="bold")
    ax1.set_xlabel("Taxa Total — Bits por Pixel (bpp)", fontsize=11)
    ax1.set_ylabel("PSNR Médio (dB)", fontsize=11)
    ax1.legend(fontsize=9.2, loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot (b): Progressão das arquiteturas em N=100k (45 imagens) + Âncora Image-GS (2.6703 bpp)
    df_arch['stab'] = df_arch['stabilize_grads'].astype(str).str.strip().str.lower()
    orig_nostab = df_arch[(df_arch['arm_arch'] == 'original') & (df_arch['stab'] == 'false')].groupby('lambda_rdo', as_index=False)[['bpp', 'psnr']].mean().sort_values('lambda_rdo', ascending=False)
    orig_stab = df_arch[(df_arch['arm_arch'] == 'original') & (df_arch['stab'] == 'true')].groupby('lambda_rdo', as_index=False)[['bpp', 'psnr']].mean().sort_values('lambda_rdo', ascending=False)
    shal_c16 = df_arch[(df_arch['arm_arch'] == 'shallow') & (df_arch['stab'] == 'true')].groupby('lambda_rdo', as_index=False)[['bpp', 'psnr']].mean().sort_values('lambda_rdo', ascending=False)

    # 44 imagens completas para a curva de C=4
    comp_imgs_c4 = [img for img in sorted(c4_all['image_name'].unique()) if len(c4_all[c4_all['image_name'] == img]) == 3]
    shal_c4 = c4_all[c4_all['image_name'].isin(comp_imgs_c4)].groupby('lambda_rdo', as_index=False)[['bpp', 'psnr']].mean().sort_values('lambda_rdo', ascending=False)

    # Ponto âncora Image-GS em N=100k (2.6703 bpp, 38.15 dB)
    ax2.scatter([2.6703], [38.15], marker='s', s=110, color='#333333', zorder=5, label='Image-GS Original (N=100k, 2,6703 bpp)')
    ax2.annotate("Image-GS (100k)\n2,6703 bpp | 38,15 dB", (2.6703, 38.15), textcoords="offset points", xytext=(-95, -28), fontsize=8.8, fontweight='bold', color='#333333')

    ax2.plot(orig_nostab['bpp'], orig_nostab['psnr'], linestyle='--', marker='s', color='#7f7f7f', linewidth=2.0, markersize=7, label='Original C=16 s/ Estab. (N=100k)')
    ax2.plot(orig_stab['bpp'], orig_stab['psnr'], linestyle='-.', marker='D', color='#1f77b4', linewidth=2.0, markersize=7, label='Original C=16 c/ Estab. (N=100k)')
    ax2.plot(shal_c16['bpp'], shal_c16['psnr'], linestyle=':', marker='^', color='#2ca02c', linewidth=2.2, markersize=7.5, label='Shallow C=16 c/ Estab. (N=100k)')
    ax2.plot(shal_c4['bpp'], shal_c4['psnr'], linestyle='-', marker='o', color='#d62728', linewidth=2.6, markersize=8, label='Shallow C=4 c/ Estab. (N=100k) [-29,9% bpp]')

    pt_c4_0005 = shal_c4[np.isclose(shal_c4['lambda_rdo'], 0.0005)].iloc[0]
    ax2.annotate(
        "", xy=(pt_c4_0005['bpp'], pt_c4_0005['psnr']), xytext=(2.6703, 38.15),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.0, ls="-.")
    )
    ax2.text(
        (2.6703 + pt_c4_0005['bpp']) / 2, (38.15 + pt_c4_0005['psnr']) / 2 - 0.55,
        "-29,9% bpp (λ=0,0005)", fontsize=9.5, fontweight='bold', color='#b01515', ha='center'
    )

    ax2.set_title("(b) Evolução das Etapas em N=100k (45 Imagens) vs. Image-GS", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("Taxa Total — Bits por Pixel (bpp)", fontsize=11)
    ax2.set_ylabel("PSNR Médio (dB)", fontsize=11)
    ax2.legend(fontsize=9.0, loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig2.suptitle("Desempenho em N = 100.000 Gaussianas: Image-GS Original vs. GaussianARM (Shallow, C=4)", fontsize=14.5, fontweight="bold", y=1.02)
    fig2.tight_layout()
    for d in (out_dir, fig_dir):
        fig2.savefig(os.path.join(d, "rd_psnr_shallow_c4_vs_baseline_100k.pdf"), bbox_inches="tight")
        fig2.savefig(os.path.join(d, "rd_psnr_shallow_c4_vs_baseline_100k.png"), dpi=300, bbox_inches="tight")
    fig2.savefig(os.path.join(art_dir, "rd_psnr_shallow_c4_vs_baseline_100k.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print("Gráficos gerados com sucesso!")


if __name__ == "__main__":
    main()
