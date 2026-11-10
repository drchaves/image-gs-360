import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("resultados_experimentos.csv")

def plot_image(img_name, filename, title):
    df_img = df[df['image_name'] == img_name]
    anchor = df_img[df_img['lambda_rdo'] == 0.0].sort_values(by='bpp')
    prop = df_img[df_img['lambda_rdo'] == 0.0005].sort_values(by='bpp')
    
    plt.figure(figsize=(8, 6))
    
    # Plot Anchor
    plt.plot(anchor['bpp'], anchor['psnr'], marker='o', linestyle='-', color='red', label='Image-GS Original (Âncora)')
    
    # Plot Proposed
    plt.plot(prop['bpp'], prop['psnr'], marker='s', linestyle='-', color='blue', label='GaussianARM (Proposto, λ=0.0005)')
    
    plt.title(title)
    plt.xlabel("BPP (Bits Per Pixel) - Menor é melhor")
    plt.ylabel("PSNR (dB) - Maior é melhor")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

# Plot Anime (Sucesso)
plot_image('anime-2_2k', 'curva_anime.png', 'Curva Taxa-Distorção (RDO) - Categoria: Anime (Sucesso)')

# Plot Foto (Colapso)
plot_image('photo-1_2k', 'curva_photo.png', 'Curva Taxa-Distorção (RDO) - Categoria: Fotografia (Colapso)')
