import csv
import glob
import os
import re
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


def extract_ws_psnr_from_logs(row):
    if "ws_psnr" in row and row["ws_psnr"] not in ("", None):
        return float(row["ws_psnr"])
    img = row["image_name"]
    arch = row["arm_arch"]
    stab = "_stab" if str(row["stabilize_grads"]).strip().lower() == "true" else ""
    num_g = row["num_gaussians"]
    lmb = row["lambda_rdo"]
    exp_name = f"exp_{img}_{arch}{stab}_g{num_g}_L{lmb}"
    logs = glob.glob(f"results/{exp_name}/**/log_train.txt", recursive=True)
    if logs:
        txt = open(logs[-1], encoding="utf-8", errors="ignore").read()
        matches = re.findall(r"WS-PSNR:\s*([0-9.]+)", txt)
        if matches:
            return float(matches[-1])
    return float(row["psnr"])


def main():
    csv_path = "/home/diego/image-gs/resultados_diego_csvs/resultados_arm_archs_stabilize_grads_lambda_espacado.csv"
    df = pd.read_csv(csv_path)

    # Garante que a coluna ws_psnr exista no DataFrame (e salva no CSV se ainda não existir)
    if "ws_psnr" not in df.columns:
        ws_vals = [extract_ws_psnr_from_logs(row) for _, row in df.iterrows()]
        psnr_idx = list(df.columns).index("psnr") + 1
        df.insert(psnr_idx, "ws_psnr", [f"{v:.2f}" for v in ws_vals])
        df.to_csv(csv_path, index=False)
        df["ws_psnr"] = ws_vals
    else:
        df["ws_psnr"] = df["ws_psnr"].astype(float)

    df["bpp"] = df["bpp"].astype(float)
    df["psnr"] = df["psnr"].astype(float)
    df["stabilize_grads_bool"] = df["stabilize_grads"].astype(str).str.strip().str.lower() == "true"

    out_dir = "plots_arm_arch"
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

    arch_colors = {
        "original": "#1b7a5a",  # Verde
        "shallow":  "#c0601a",  # Laranja
        "deep":     "#5a4bb5",  # Roxo
    }
    arch_markers = {
        "original": "o",
        "shallow":  "s",
        "deep":     "^",
    }

    images = list(df["image_name"].unique())
    archs = ["original", "shallow", "deep"]

    print("==========================================================================")
    print(" CÁLCULO DE BD-RATE COM WS-PSNR (vs. original | stabilize_grads=False)")
    print("==========================================================================")

    # Figura em grid 2x3 (5 imagens + Média Geral)
    fig_grid, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    axes = axes.flatten()

    for idx, img in enumerate(images):
        df_img = df[df["image_name"] == img]
        df_anchor = df_img[(df_img["arm_arch"] == "original") & (~df_img["stabilize_grads_bool"])].sort_values("ws_psnr")
        r_base = df_anchor["bpp"].values
        p_base = df_anchor["ws_psnr"].values

        print(f"\nImagem: {img}")
        fig_single, ax_s = plt.subplots(figsize=(8, 5.8))

        for arch in archs:
            for stab in [False, True]:
                sub = df_img[(df_img["arm_arch"] == arch) & (df_img["stabilize_grads_bool"] == stab)].sort_values("lambda_rdo", ascending=False)
                if sub.empty:
                    continue
                sub_p = sub.sort_values("ws_psnr")
                if arch == "original" and not stab:
                    bd_str = "Âncora"
                else:
                    bd_val = bjontegaard_rate(r_base, p_base, sub_p["bpp"].values, sub_p["ws_psnr"].values)
                    bd_str = f"BD: {bd_val:+.2f}%" if not np.isnan(bd_val) else "BD: N/A"
                    print(f"  -> {arch:8s} | stab={str(stab):5s} : {bd_str}")

                ls = "-" if stab else "--"
                alpha = 1.0 if stab else 0.72
                mfc = arch_colors[arch] if stab else "white"
                stab_lbl = "com stab" if stab else "sem stab"
                lbl = f"{arch} ({stab_lbl}) [{bd_str}]"

                for ax in (ax_s, axes[idx]):
                    ax.plot(
                        sub["bpp"], sub["ws_psnr"],
                        linestyle=ls, marker=arch_markers[arch],
                        color=arch_colors[arch], markerfacecolor=mfc,
                        markeredgecolor=arch_colors[arch], markeredgewidth=1.8,
                        linewidth=2.2 if stab else 1.6, markersize=7, alpha=alpha,
                        label=lbl
                    )

        for ax, is_single in ((ax_s, True), (axes[idx], False)):
            ax.set_title(f"Taxa-Distorção (WS-PSNR vs. BPP) — {img}" if is_single else f"{img}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Taxa — Bits por Pixel (BPP)", fontsize=11)
            ax.set_ylabel("WS-PSNR (dB)", fontsize=11)
            ax.legend(fontsize=8.5 if is_single else 7.8, loc="best", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)

        single_png = os.path.join(out_dir, f"rd_ws_psnr_{img}.png")
        single_pdf = os.path.join(out_dir, f"rd_ws_psnr_{img}.pdf")
        fig_single.savefig(single_png, dpi=300, bbox_inches="tight")
        fig_single.savefig(single_pdf, bbox_inches="tight")
        plt.close(fig_single)

    # 6º painel e figura isolada: Curva Média Global (agrupada por arch, stabilize_grads, lambda_rdo)
    df_mean = df.groupby(["arm_arch", "stabilize_grads_bool", "lambda_rdo"], as_index=False)[["bpp", "ws_psnr", "psnr"]].mean()
    df_mean_anchor = df_mean[(df_mean["arm_arch"] == "original") & (~df_mean["stabilize_grads_bool"])].sort_values("ws_psnr")
    r_mbase = df_mean_anchor["bpp"].values
    p_mbase = df_mean_anchor["ws_psnr"].values

    fig_mean, ax_m = plt.subplots(figsize=(8.5, 6.0))
    ax_g = axes[5]

    print("\n--------------------------------------------------------------------------")
    print(" CURVA MÉDIA GLOBAL (Média das 5 imagens por lambda):")
    for arch in archs:
        for stab in [False, True]:
            sub = df_mean[(df_mean["arm_arch"] == arch) & (df_mean["stabilize_grads_bool"] == stab)].sort_values("lambda_rdo", ascending=False)
            sub_p = sub.sort_values("ws_psnr")
            if arch == "original" and not stab:
                bd_str = "Âncora"
            else:
                bd_val = bjontegaard_rate(r_mbase, p_mbase, sub_p["bpp"].values, sub_p["ws_psnr"].values)
                bd_str = f"BD: {bd_val:+.2f}%" if not np.isnan(bd_val) else "BD: N/A"
                print(f"  * {arch:8s} | stab={str(stab):5s} -> {bd_str}")

            ls = "-" if stab else "--"
            alpha = 1.0 if stab else 0.72
            mfc = arch_colors[arch] if stab else "white"
            stab_lbl = "com stab" if stab else "sem stab"
            lbl = f"{arch} ({stab_lbl}) [{bd_str}]"

            for ax in (ax_m, ax_g):
                ax.plot(
                    sub["bpp"], sub["ws_psnr"],
                    linestyle=ls, marker=arch_markers[arch],
                    color=arch_colors[arch], markerfacecolor=mfc,
                    markeredgecolor=arch_colors[arch], markeredgewidth=1.8,
                    linewidth=2.3 if stab else 1.7, markersize=7.5, alpha=alpha,
                    label=lbl
                )

    for ax, is_single in ((ax_m, True), (ax_g, False)):
        ax.set_title("Taxa-Distorção Média Global (WS-PSNR vs. BPP)" if is_single else "Média Global (5 imagens)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Taxa — Bits por Pixel (BPP)", fontsize=11)
        ax.set_ylabel("WS-PSNR (dB)", fontsize=11)
        ax.legend(fontsize=9.0 if is_single else 7.8, loc="best", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    mean_png = os.path.join(out_dir, "rd_ws_psnr_media_geral.png")
    mean_pdf = os.path.join(out_dir, "rd_ws_psnr_media_geral.pdf")
    fig_mean.savefig(mean_png, dpi=300, bbox_inches="tight")
    fig_mean.savefig(mean_pdf, bbox_inches="tight")
    plt.close(fig_mean)

    fig_grid.suptitle("Curvas Taxa-Distorção (WS-PSNR vs. BPP) — Arquiteturas do ARM com e sem Estabilização de Gradientes", fontsize=16, fontweight="bold", y=1.01)
    fig_grid.tight_layout()
    grid_png = os.path.join(out_dir, "rd_ws_psnr_grid_completo.png")
    grid_pdf = os.path.join(out_dir, "rd_ws_psnr_grid_completo.pdf")
    fig_grid.savefig(grid_png, dpi=300, bbox_inches="tight")
    fig_grid.savefig(grid_pdf, bbox_inches="tight")
    plt.close(fig_grid)

    print("==========================================================================")
    print(f"Gráficos salvos em '{out_dir}/' (.png em 300 dpi e .pdf):")
    print(f"  - {grid_png}")
    print(f"  - {mean_png}")
    print("  - rd_ws_psnr_<imagem>.png / .pdf")


if __name__ == "__main__":
    main()
