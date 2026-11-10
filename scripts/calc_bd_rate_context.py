import csv
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def bjontegaard_rate(rate1, psnr1, rate2, psnr2):
    log_rate1 = np.log10(rate1)
    log_rate2 = np.log10(rate2)
    try:
        deg1 = 2 if (len(psnr1) >= 3 and np.min(np.abs(np.diff(psnr1))) >= 0.02) else 1
        deg2 = 2 if (len(psnr2) >= 3 and np.min(np.abs(np.diff(psnr2))) >= 0.02) else 1
        p1 = np.polyfit(psnr1, log_rate1, deg1)
        p2 = np.polyfit(psnr2, log_rate2, deg2)
    except Exception:
        return float('nan')

    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    if max_psnr <= min_psnr:
        return float('nan')

    int_p1 = np.polyint(p1)
    int_p2 = np.polyint(p2)
    int1 = np.polyval(int_p1, max_psnr) - np.polyval(int_p1, min_psnr)
    int2 = np.polyval(int_p2, max_psnr) - np.polyval(int_p2, min_psnr)
    avg_diff = (int2 - int1) / (max_psnr - min_psnr)
    return (10**avg_diff - 1) * 100


def get_curve_points(df, img, ctx, metric='psnr', rate_col='bpp'):
    sub = df[(df['image_name'] == img) & (df['arm_context'] == ctx)]
    if len(sub) < 3:
        return None, None
    sub_sorted = sub.sort_values(metric)
    return sub_sorted[rate_col].values, sub_sorted[metric].values


def main():
    base_dir = "/home/diego/image-gs"
    csv_path = os.path.join(base_dir, "resultados_diego_csvs/results_shallow_stab_context_4_8_16_32.csv")
    df = pd.read_csv(csv_path)

    df['arm_context'] = df['arm_context'].astype(int)
    df['lambda_rdo'] = df['lambda_rdo'].astype(float)
    for col in ['psnr', 'ssim', 'msssim', 'lpips', 'bytes', 'bpp_img', 'bpp_arm', 'bpp']:
        df[col] = df[col].astype(float)
    df['category'] = df['image_name'].apply(lambda x: x.split('-')[0])

    contexts = [4, 8, 16, 32]
    images = sorted(df['image_name'].unique())
    complete_images = [
        img for img in images
        if all(len(df[(df['image_name'] == img) & (df['arm_context'] == c)]) == 3 for c in contexts)
    ]

    print(f"Total de imagens no dataset: {len(images)} | Imagens 100% completas nos 4 contextos: {len(complete_images)}")

    df_comp = df[df['image_name'].isin(complete_images)]
    print("\n=======================================================================================")
    print(" MÉDIAS GLOBAIS POR CONTEXTO E LAMBDA (44 imagens completas em todos os contextos)")
    print("=======================================================================================")
    summary_rows = []
    for ctx in contexts:
        for lmb in sorted(df['lambda_rdo'].unique()):
            sub = df_comp[(df_comp['arm_context'] == ctx) & (np.isclose(df_comp['lambda_rdo'], lmb))]
            summary_rows.append({
                'C': ctx,
                'lambda': lmb,
                'BPP_total': sub['bpp'].mean(),
                'BPP_img': sub['bpp_img'].mean(),
                'BPP_arm': sub['bpp_arm'].mean(),
                'PSNR': sub['psnr'].mean(),
                'MS_SSIM': sub['msssim'].mean(),
                'LPIPS': sub['lpips'].mean(),
            })
    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Média Consolidada por Contexto (média dos 3 lambdas nas 44 imagens) ---")
    df_ctx_mean = df_comp.groupby('arm_context')[['bpp', 'bpp_img', 'bpp_arm', 'psnr', 'msssim', 'lpips']].mean()
    print(df_ctx_mean.to_string(float_format=lambda x: f"{x:.4f}"))

    # 2. Cálculo de BD-Rate com PSNR
    for anchor_ctx in [16, 4]:
        for rate_col, rate_lbl in [('bpp', 'BPP Total (Imagem + ARM)'), ('bpp_img', 'BPP Imagem (Sem Overhead do ARM)')]:
            print("\n=======================================================================================")
            print(f" BD-RATE (PSNR vs. {rate_lbl}) | ÂNCORA: Contexto C={anchor_ctx}")
            print("=======================================================================================")

            df_mean_curve = df_comp.groupby(['arm_context', 'lambda_rdo'], as_index=False)[['bpp', 'bpp_img', 'psnr']].mean()
            anc_sub = df_mean_curve[df_mean_curve['arm_context'] == anchor_ctx].sort_values('psnr')
            r_anc_mean = anc_sub[rate_col].values
            p_anc_mean = anc_sub['psnr'].values

            print("\n[A] BD-Rate calculado sobre a Curva Média Global (44 imagens completas):")
            for ctx in contexts:
                if ctx == anchor_ctx:
                    print(f"  * Contexto C={ctx:<2d} :   0.00% (Âncora)")
                    continue
                sub = df_mean_curve[df_mean_curve['arm_context'] == ctx].sort_values('psnr')
                bd_mean_curve = bjontegaard_rate(r_anc_mean, p_anc_mean, sub[rate_col].values, sub['psnr'].values)
                print(f"  * Contexto C={ctx:<2d} : {bd_mean_curve:+6.2f}% (PSNR)")

            print("\n[B] BD-Rate por Categoria (Curva Média de cada Categoria):")
            for cat in sorted(df_comp['category'].unique()):
                df_cat = df_comp[df_comp['category'] == cat].groupby(['arm_context', 'lambda_rdo'], as_index=False)[[rate_col, 'psnr']].mean()
                anc_c = df_cat[df_cat['arm_context'] == anchor_ctx].sort_values('psnr')
                cat_bds = []
                for ctx in contexts:
                    if ctx == anchor_ctx:
                        continue
                    sub_c = df_cat[df_cat['arm_context'] == ctx].sort_values('psnr')
                    bd_c = bjontegaard_rate(anc_c[rate_col].values, anc_c['psnr'].values, sub_c[rate_col].values, sub_c['psnr'].values)
                    cat_bds.append(f"C={ctx}: {bd_c:+.2f}%")
                print(f"  - {cat:<10}: " + " | ".join(cat_bds))

            print("\n[C] BD-Rate calculado Imagem por Imagem (Média / Mediana):")
            for ctx in contexts:
                if ctx == anchor_ctx:
                    continue
                bds_all = []
                for img in images:
                    r_base, p_base = get_curve_points(df, img, anchor_ctx, metric='psnr', rate_col=rate_col)
                    r_comp, p_comp = get_curve_points(df, img, ctx, metric='psnr', rate_col=rate_col)
                    if r_base is None or r_comp is None:
                        continue
                    bd = bjontegaard_rate(r_base, p_base, r_comp, p_comp)
                    if not np.isnan(bd):
                        bds_all.append(bd)

                filt = [x for x in bds_all if abs(x) < 100]
                print(f"  * Contexto C={ctx:<2d} vs C={anchor_ctx:<2d} ({len(bds_all)} imgs): "
                      f"Média={np.mean(bds_all):+6.2f}% | Mediana={np.median(bds_all):+6.2f}% | "
                      f"Média s/ outliers (<100%)={np.mean(filt):+6.2f}% ({len(filt)}/{len(bds_all)})")

    # ==========================================
    # GERAÇÃO DE GRÁFICOS (PSNR x BPP)
    # ==========================================
    out_dir = os.path.join(base_dir, "plots_arm_context")
    os.makedirs(out_dir, exist_ok=True)

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

    ctx_colors = {
        4:  "#1f77b4",  # Azul
        8:  "#2ca02c",  # Verde
        16: "#d62728",  # Vermelho (Padrão/Âncora)
        32: "#9467bd",  # Roxo
    }
    ctx_markers = {
        4:  "o",
        8:  "s",
        16: "D",
        32: "^",
    }
    ctx_linestyles = {
        4:  "--",
        8:  "-.",
        16: "-",
        32: ":",
    }

    # --- GRÁFICO 1: Curva Média Global (BPP Total vs BPP Imagem lado a lado) ---
    df_mean_curve = df_comp.groupby(['arm_context', 'lambda_rdo'], as_index=False)[['bpp', 'bpp_img', 'bpp_arm', 'psnr']].mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, (rate_col, title_suffix, xlabel) in zip(axes, [
        ('bpp', 'Taxa Total (Imagem + Pesos do ARM)', 'Taxa Total — Bits por Pixel (bpp)'),
        ('bpp_img', 'Apenas Imagem (Sem Overhead do ARM)', 'Taxa das Gaussianas — bpp (bpp_img)')
    ]):
        anc_sub = df_mean_curve[df_mean_curve['arm_context'] == 16].sort_values('psnr')
        r_anc = anc_sub[rate_col].values
        p_anc = anc_sub['psnr'].values

        for ctx in contexts:
            sub = df_mean_curve[df_mean_curve['arm_context'] == ctx].sort_values('lambda_rdo', ascending=False)
            sub_p = sub.sort_values('psnr')
            if ctx == 16:
                bd_str = "Âncora"
            else:
                bd_val = bjontegaard_rate(r_anc, p_anc, sub_p[rate_col].values, sub_p['psnr'].values)
                bd_str = f"BD-Rate: {bd_val:+.2f}%"

            arm_ovh = sub['bpp_arm'].iloc[0]
            lbl = f"C={ctx} (bpp_arm={arm_ovh:.4f}) [{bd_str}]"
            ax.plot(
                sub[rate_col], sub['psnr'],
                linestyle=ctx_linestyles[ctx],
                marker=ctx_markers[ctx],
                color=ctx_colors[ctx],
                linewidth=2.4 if ctx in (8, 16) else 2.0,
                markersize=8,
                label=lbl
            )
            if ctx == 16:
                for _, row in sub.iterrows():
                    ax.annotate(
                        f"λ={row['lambda_rdo']:g}",
                        (row[rate_col], row['psnr']),
                        textcoords="offset points", xytext=(6, -10),
                        fontsize=8.5, color=ctx_colors[16], fontweight='bold'
                    )

        ax.set_title(f"PSNR vs. bpp — {title_suffix}", fontsize=12.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9.5, loc="lower right", frameon=True)

    axes[0].set_ylabel("PSNR Médio (dB)", fontsize=11)
    fig.suptitle("Ablação de Contexto do ARM (Shallow + Stabilize Grads) — Média Global (45 imagens)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    p1_png = os.path.join(out_dir, "rd_psnr_media_global_total_vs_img.png")
    p1_pdf = os.path.join(out_dir, "rd_psnr_media_global_total_vs_img.pdf")
    fig.savefig(p1_png, dpi=300, bbox_inches="tight")
    fig.savefig(p1_pdf, bbox_inches="tight")
    plt.close(fig)

    # --- GRÁFICO 2: Curva Média Global Isolada (BPP Total) ---
    fig_s, ax_s = plt.subplots(figsize=(8.5, 6.0))
    anc_sub = df_mean_curve[df_mean_curve['arm_context'] == 16].sort_values('psnr')
    r_anc = anc_sub['bpp'].values
    p_anc = anc_sub['psnr'].values

    for ctx in contexts:
        sub = df_mean_curve[df_mean_curve['arm_context'] == ctx].sort_values('lambda_rdo', ascending=False)
        sub_p = sub.sort_values('psnr')
        if ctx == 16:
            bd_str = "Âncora (C=16)"
        else:
            bd_val = bjontegaard_rate(r_anc, p_anc, sub_p['bpp'].values, sub_p['psnr'].values)
            bd_str = f"BD-Rate: {bd_val:+.2f}%"

        lbl = f"Contexto C={ctx} [{bd_str}]"
        ax_s.plot(
            sub['bpp'], sub['psnr'],
            linestyle=ctx_linestyles[ctx],
            marker=ctx_markers[ctx],
            color=ctx_colors[ctx],
            linewidth=2.4,
            markersize=8,
            label=lbl
        )
    ax_s.set_title("Taxa-Distorção Média Global (45 imagens) — PSNR vs. bpp", fontsize=13, fontweight="bold")
    ax_s.set_xlabel("Taxa Total — Bits por Pixel (bpp)", fontsize=11)
    ax_s.set_ylabel("PSNR Médio (dB)", fontsize=11)
    ax_s.legend(fontsize=10, loc="lower right", frameon=True)
    ax_s.grid(True, linestyle="--", alpha=0.6)
    p2_png = os.path.join(out_dir, "rd_psnr_media_global.png")
    p2_pdf = os.path.join(out_dir, "rd_psnr_media_global.pdf")
    fig_s.savefig(p2_png, dpi=300, bbox_inches="tight")
    fig_s.savefig(p2_pdf, bbox_inches="tight")
    plt.close(fig_s)

    # --- GRÁFICO 3: Painel 2x3 por Categoria (5 categorias + Média Global) ---
    categories = sorted(df_comp['category'].unique())
    fig_cat, axes_cat = plt.subplots(2, 3, figsize=(18, 10.5))
    axes_cat = axes_cat.flatten()

    for idx, cat in enumerate(categories):
        ax = axes_cat[idx]
        df_cat = df_comp[df_comp['category'] == cat].groupby(['arm_context', 'lambda_rdo'], as_index=False)[['bpp', 'psnr']].mean()
        anc_cat = df_cat[df_cat['arm_context'] == 16].sort_values('psnr')
        r_anc_c = anc_cat['bpp'].values
        p_anc_c = anc_cat['psnr'].values

        for ctx in contexts:
            sub = df_cat[df_cat['arm_context'] == ctx].sort_values('lambda_rdo', ascending=False)
            sub_p = sub.sort_values('psnr')
            if ctx == 16:
                bd_str = "Âncora"
            else:
                bd_val = bjontegaard_rate(r_anc_c, p_anc_c, sub_p['bpp'].values, sub_p['psnr'].values)
                bd_str = f"BD: {bd_val:+.2f}%"

            ax.plot(
                sub['bpp'], sub['psnr'],
                linestyle=ctx_linestyles[ctx],
                marker=ctx_markers[ctx],
                color=ctx_colors[ctx],
                linewidth=2.2,
                markersize=7,
                label=f"C={ctx} [{bd_str}]"
            )
        cat_pt = {"anime": "Anime", "art": "Arte", "painting": "Pintura", "photo": "Fotografia", "vector": "Vetorial"}.get(cat, cat.capitalize())
        ax.set_title(f"Categoria: {cat_pt} (9 imagens)", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Taxa Total — bpp", fontsize=10.5)
        ax.set_ylabel("PSNR (dB)", fontsize=10.5)
        ax.legend(fontsize=8.8, loc="lower right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    # 6º subplot: Média Global
    ax_g = axes_cat[5]
    anc_sub = df_mean_curve[df_mean_curve['arm_context'] == 16].sort_values('psnr')
    for ctx in contexts:
        sub = df_mean_curve[df_mean_curve['arm_context'] == ctx].sort_values('lambda_rdo', ascending=False)
        sub_p = sub.sort_values('psnr')
        if ctx == 16:
            bd_str = "Âncora"
        else:
            bd_val = bjontegaard_rate(anc_sub['bpp'].values, anc_sub['psnr'].values, sub_p['bpp'].values, sub_p['psnr'].values)
            bd_str = f"BD: {bd_val:+.2f}%"
        ax_g.plot(
            sub['bpp'], sub['psnr'],
            linestyle=ctx_linestyles[ctx],
            marker=ctx_markers[ctx],
            color=ctx_colors[ctx],
            linewidth=2.4,
            markersize=7.5,
            label=f"C={ctx} [{bd_str}]"
        )
    ax_g.set_title("Média Global (45 imagens)", fontsize=12.5, fontweight="bold")
    ax_g.set_xlabel("Taxa Total — bpp", fontsize=10.5)
    ax_g.set_ylabel("PSNR (dB)", fontsize=10.5)
    ax_g.legend(fontsize=8.8, loc="lower right", frameon=True)
    ax_g.grid(True, linestyle="--", alpha=0.6)

    fig_cat.suptitle("Curvas Taxa-Distorção (PSNR vs. bpp) por Categoria — Ablação de Contexto do ARM (C = 4, 8, 16, 32)", fontsize=15, fontweight="bold", y=1.01)
    fig_cat.tight_layout()
    p3_png = os.path.join(out_dir, "rd_psnr_por_categoria.png")
    p3_pdf = os.path.join(out_dir, "rd_psnr_por_categoria.pdf")
    fig_cat.savefig(p3_png, dpi=300, bbox_inches="tight")
    fig_cat.savefig(p3_pdf, bbox_inches="tight")
    plt.close(fig_cat)

    # --- GRÁFICO 4: Painel das Arquiteturas + Stabilize Grads com PSNR normal ---
    stab_csv = os.path.join(base_dir, "resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv")
    if os.path.exists(stab_csv):
        df_st = pd.read_csv(stab_csv)
        df_st['stabilize_grads_str'] = df_st['stabilize_grads'].astype(str).str.strip().str.lower()
        for col in ['bpp', 'psnr', 'lambda_rdo']:
            df_st[col] = df_st[col].astype(float)
        st_imgs = sorted(df_st['image_name'].unique())
        comp_st_imgs = [img for img in st_imgs if len(df_st[df_st['image_name'] == img]) == 18]
        df_stc = df_st[df_st['image_name'].isin(comp_st_imgs)]
        df_stm = df_stc.groupby(['arm_arch', 'stabilize_grads_str', 'lambda_rdo'], as_index=False)[['bpp', 'psnr']].mean()

        fig_arch, axes_arch = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for ax, arch in zip(axes_arch, ['original', 'shallow', 'deep']):
            b_sub = df_stm[(df_stm['arm_arch'] == arch) & (df_stm['stabilize_grads_str'] == 'false')].sort_values('lambda_rdo', ascending=False)
            s_sub = df_stm[(df_stm['arm_arch'] == arch) & (df_stm['stabilize_grads_str'] == 'true')].sort_values('lambda_rdo', ascending=False)
            bd_st = bjontegaard_rate(
                b_sub.sort_values('psnr')['bpp'].values, b_sub.sort_values('psnr')['psnr'].values,
                s_sub.sort_values('psnr')['bpp'].values, s_sub.sort_values('psnr')['psnr'].values
            )
            ax.plot(b_sub['bpp'], b_sub['psnr'], linestyle='--', marker='s', color='#1f77b4', linewidth=2.0, markersize=7, label='Sem Estabilização (Âncora)')
            ax.plot(s_sub['bpp'], s_sub['psnr'], linestyle='-', marker='o', color='#ff7f0e', linewidth=2.3, markersize=7, label=f'Com Estabilização [BD: {bd_st:+.2f}%]')
            ax.set_title(f"Arquitetura: {arch.capitalize()}", fontsize=12.5, fontweight="bold")
            ax.set_xlabel("Taxa Total — bpp", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=9.2, loc="lower right", frameon=True)
        axes_arch[0].set_ylabel("PSNR Médio (dB)", fontsize=11)
        fig_arch.suptitle("Impacto da Estabilização de Gradientes por Arquitetura (PSNR vs. bpp — 45 imagens)", fontsize=14, fontweight="bold", y=1.02)
        fig_arch.tight_layout()
        pa_png = os.path.join(out_dir, "rd_psnr_painel_arquiteturas_stabilize.png")
        pa_pdf = os.path.join(out_dir, "rd_psnr_painel_arquiteturas_stabilize.pdf")
        fig_arch.savefig(pa_png, dpi=300, bbox_inches="tight")
        fig_arch.savefig(pa_pdf, bbox_inches="tight")
        plt.close(fig_arch)

    # --- GRÁFICO 5: Gráficos individuais por imagem (PSNR) ---
    indiv_dir = os.path.join(out_dir, "por_imagem_psnr")
    os.makedirs(indiv_dir, exist_ok=True)
    for img in images:
        fig_i, ax_i = plt.subplots(figsize=(7.5, 5.2))
        df_img = df[df['image_name'] == img]
        r_anc_i, p_anc_i = get_curve_points(df, img, 16, metric='psnr', rate_col='bpp')

        for ctx in contexts:
            sub = df_img[df_img['arm_context'] == ctx].sort_values('lambda_rdo', ascending=False)
            if sub.empty:
                continue
            r_c, p_c = get_curve_points(df, img, ctx, metric='psnr', rate_col='bpp')
            if ctx == 16:
                bd_str = "Âncora"
            elif r_anc_i is not None and r_c is not None:
                bd_val = bjontegaard_rate(r_anc_i, p_anc_i, r_c, p_c)
                bd_str = f"BD: {bd_val:+.2f}%" if not np.isnan(bd_val) else "BD: N/A"
            else:
                bd_str = f"{len(sub)} pts (incompleto)"

            ax_i.plot(
                sub['bpp'], sub['psnr'],
                linestyle=ctx_linestyles[ctx],
                marker=ctx_markers[ctx],
                color=ctx_colors[ctx],
                linewidth=2.0,
                markersize=7,
                label=f"C={ctx} [{bd_str}]"
            )
        ax_i.set_title(f"Taxa-Distorção (PSNR vs. bpp) — {img}", fontsize=12, fontweight="bold")
        ax_i.set_xlabel("Taxa Total — Bits por Pixel (bpp)", fontsize=10.5)
        ax_i.set_ylabel("PSNR (dB)", fontsize=10.5)
        ax_i.legend(fontsize=9, loc="lower right", frameon=True)
        ax_i.grid(True, linestyle="--", alpha=0.6)
        fig_i.savefig(os.path.join(indiv_dir, f"rd_psnr_{img}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig_i)

    print("\n=======================================================================================")
    print(f"Gráficos (PSNR) salvos em: {out_dir}")
    print(f"  1. {p1_png}")
    print(f"  2. {p2_png}")
    print(f"  3. {p3_png}")
    print(f"  4. {pa_png}")
    print(f"  5. {len(images)} gráficos individuais em {indiv_dir}/")
    print("=======================================================================================")


if __name__ == "__main__":
    main()
