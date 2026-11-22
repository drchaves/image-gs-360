"""Gera a figura da Avaliação Visual Qualitativa (painel 4x4) para as 4 imagens escolhidas:
  1. art-2_2k       (Melhor resultado — Dataset Principal)
  2. photo-6_2k     (Resultado ruim   — Dataset Principal)
  3. 50d13125...b91 (Melhor resultado — CLIC 2025)
  4. 2a760bf1...c12 (Resultado ruim   — CLIC 2025)

Colunas:
  (a) Imagem Original (Ground Truth)
  (b) Reconstrução Image-GS Original (Baseline, N=100k)
  (c) Nossa Reconstrução (GaussianARM Shallow, C=2, N=100k, lambda=0.0001)
  (d) Mapa de Erro Absoluto |X - X_hat| (amplificado 10x para visualização)
"""
import glob
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_render(prefix_list):
    for pat in prefix_list:
        matches = sorted(glob.glob(f"results/{pat}/*/render_res-*.jpg") + glob.glob(f"results/{pat}/*/render_res-*.png"))
        if matches:
            return matches[-1]
    return None


def load_rgb(path, target_size=None):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if target_size is not None and im.size != target_size:
            im = im.resize(target_size, Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.float32) / 255.0


def main():
    base_dir = "/home/diego/image-gs"
    os.chdir(base_dir)
    fig_dir = os.path.join(base_dir, "tcc_diego/figuras")
    art_dir = "/home/diego/.gemini/antigravity/brain/59567a6c-0bcf-496f-ba44-271dbb78ea6a"
    os.makedirs(fig_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    })

    cases = [
        {
            "title": "Conj. Principal (Sucesso): art-2_2k\nBD-Rate: -39,14% | Δbpp: -34,0%",
            "gt": "media/images/art-2_2k.png",
            "base": ["final_art-2_2k_base_g100000_L0"],
            "ours": ["final_art-2_2k_shc2stab_g100000_L0.0001", "final_art-2_2k_shc2stab_g100000_L1e-04", "final_art-2_2k_shc2stab_g100000_L0.0005"],
            "base_lbl": "44,92 dB | 2,6703 bpp",
            "ours_lbl": "44,27 dB | 1,7619 bpp (-34,0%)",
        },
        {
            "title": "Conj. Principal (Falha): photo-6_2k\nBD-Rate: +20,10% | Δbpp: -33,4%",
            "gt": "media/images/photo-6_2k.png",
            "base": ["final_photo-6_2k_base_g100000_L0"],
            "ours": ["final_photo-6_2k_shc2stab_g100000_L0.0001", "final_photo-6_2k_shc2stab_g100000_L1e-04", "final_photo-6_2k_shc2stab_g100000_L0.0005"],
            "base_lbl": "47,37 dB | 2,6703 bpp",
            "ours_lbl": "46,79 dB | 1,7788 bpp (-33,4%)",
        },
        {
            "title": "CLIC 2025 (Sucesso): 50d13125...b91\nBD-Rate: -31,09% | Δbpp: -26,8%",
            "gt": "media/clic2025_image_test/50d13125b665e0c4c1673264807bfc3cc7a217819329696843fea4c69aff8b91.png",
            "base": ["clic2025_50d13125*_base_g100000_L0"],
            "ours": ["clic2025_50d13125*_shc2stab_g100000_L0.0001", "clic2025_50d13125*_shc2stab_g100000_L1e-04"],
            "base_lbl": "40,04 dB | 4,0211 bpp",
            "ours_lbl": "40,30 dB | 2,9419 bpp (-26,8%)",
        },
        {
            "title": "CLIC 2025 (Falha): 2a760bf1...c12\nBD-Rate: +17,94% | Δbpp: -32,8%",
            "gt": "media/clic2025_image_test/2a760bf1ccea6998be3890dcc2371b23f35fe3006562884b45bc534a621d5c12.png",
            "base": ["clic2025_2a760bf1*_base_g100000_L0"],
            "ours": ["clic2025_2a760bf1*_shc2stab_g100000_L0.0001", "clic2025_2a760bf1*_shc2stab_g100000_L1e-04"],
            "base_lbl": "43,48 dB | 5,8930 bpp",
            "ours_lbl": "43,37 dB | 3,9594 bpp (-32,8%)",
        },
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16.5, 16.0))
    col_titles = [
        "(a) Imagem Original (Ground Truth)",
        "(b) Image-GS Original (Baseline)",
        "(c) Método Proposto (Shallow, C=2)",
        "(d) Mapa de Erro Absoluto |X - X̂| (×10)",
    ]

    for r_idx, c in enumerate(cases):
        b_path = find_render(c["base"])
        o_path = find_render(c["ours"])
        print(f"Linha {r_idx+1}: base={b_path} | ours={o_path}")

        gt_img = load_rgb(c["gt"], target_size=None)
        h, w = gt_img.shape[:2]
        # Reduz resolução de exibição para deixar o PDF leve sem perder nitidez
        scale_f = min(1.0, 900.0 / max(h, w))
        disp_size = (int(w * scale_f), int(h * scale_f))
        gt_disp = load_rgb(c["gt"], target_size=disp_size)
        base_disp = load_rgb(b_path, target_size=disp_size)
        ours_disp = load_rgb(o_path, target_size=disp_size)

        # Erro absoluto médio por pixel (amplificado 10x para visualização clara)
        err_map = np.mean(np.abs(gt_disp - ours_disp), axis=-1)
        err_mae = float(np.mean(err_map))

        imgs_row = [gt_disp, base_disp, ours_disp, np.clip(err_map * 10.0, 0.0, 1.0)]
        sub_labels = [
            f"Referência ({w}×{h})",
            c["base_lbl"],
            c["ours_lbl"],
            f"MAE = {err_mae:.4f} (escala ×10)",
        ]

        for c_idx in range(4):
            ax = axes[r_idx, c_idx]
            if c_idx < 3:
                ax.imshow(imgs_row[c_idx])
            else:
                im_h = ax.imshow(imgs_row[c_idx], cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel(sub_labels[c_idx], fontsize=10, fontweight="bold", labelpad=5)
            if c_idx == 0:
                ax.set_ylabel(c["title"], fontsize=10.5, fontweight="bold", labelpad=8)

    fig.tight_layout()
    pdf_out = os.path.join(fig_dir, "comparacao_visual_4_imagens.pdf")
    png_out = os.path.join(fig_dir, "comparacao_visual_4_imagens.png")
    fig.savefig(pdf_out, dpi=220, bbox_inches="tight")
    fig.savefig(png_out, dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(art_dir, "comparacao_visual_4_imagens.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo em: {pdf_out}")


if __name__ == "__main__":
    main()
