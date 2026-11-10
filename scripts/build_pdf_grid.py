import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# Limite superior (em níveis de 0-255) da escala de cor do mapa de erro absoluto.
# Fixo entre as linhas para que as intensidades sejam comparáveis.
ERROR_VMAX = 30

def create_pdf_grid(with_colorbar=False, out_name="comparacao_visual.pdf"):
    base = "results/"
    ext = "/num-100000_inv-scale-5.0_bits-14-14-14-14_top-10_g-0.3_l1-1.0_l2-0.0_ssim-0.1_decay-1-10.0_prog"
    
    paths = [
        [
            f"{base}exp_photo-5_2k_g100000_L0.0{ext}/gt_res-2048x2048.jpg",
            f"{base}exp_photo-5_2k_g100000_L0.0{ext}/render_res-2048x2048.jpg",
            f"{base}exp_photo-5_2k_g100000_L0.0005{ext}/render_res-2048x2048.jpg"
        ],
        [
            f"{base}exp_art-1_2k_g100000_L0.0{ext}/gt_res-2048x2048.jpg",
            f"{base}exp_art-1_2k_g100000_L0.0{ext}/render_res-2048x2048.jpg",
            f"{base}exp_art-1_2k_g100000_L0.0005{ext}/render_res-2048x2048.jpg"
        ]
    ]
    
    row_labels = ["Fotografia", "Arte"]
    col_labels = ["(a)", "(b)", "(c)", "(d)"]
    
    # Create a figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for r in range(2):
        for c in range(4):
            ax = axes[r, c]
            if c < 3:
                img = mpimg.imread(paths[r][c])
                ax.imshow(img)
            else:
                # Erro absoluto |gt - render (RDO)|, média sobre os canais RGB
                gt = mpimg.imread(paths[r][0]).astype(np.float32)
                rdo = mpimg.imread(paths[r][2]).astype(np.float32)
                err = np.abs(gt - rdo).mean(axis=-1)
                err_im = ax.imshow(err, cmap='inferno', vmin=0, vmax=ERROR_VMAX)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove borders
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add column labels to the first row
            if r == 0:
                ax.set_title(col_labels[c], fontsize=24, pad=15)
                
            # Add row labels to the first column
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=28, labelpad=20)

    if with_colorbar:
        # Colorbar única (escala compartilhada entre as duas linhas) ao lado da coluna (d)
        fig.canvas.draw()
        top = axes[0, 3].get_position()
        bottom = axes[1, 3].get_position()
        cax = fig.add_axes([top.x1 + 0.01, bottom.y0, 0.012, top.y1 - bottom.y0])
        cbar = fig.colorbar(err_im, cax=cax, extend='max')
        cbar.set_label("Erro absoluto médio (níveis 0–255)", fontsize=20, labelpad=15)
        cbar.ax.tick_params(labelsize=16)

    out_path = os.path.join("tcc_diego", "figuras", out_name)
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Salvo em: {out_path}")

if __name__ == "__main__":
    create_pdf_grid()
    create_pdf_grid(with_colorbar=True, out_name="comparacao_visual_v2.pdf")
