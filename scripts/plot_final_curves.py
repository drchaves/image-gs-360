import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("resultados_experimentos.csv")

def plot_image(img_name, filename, title):
    df_img = df[df['image_name'] == img_name]
    anchor = df_img[df_img['lambda_rdo'] == 0.0].sort_values(by='bpp')
    prop = df_img[df_img['lambda_rdo'] == 0.0005].sort_values(by='bpp')
    
    if len(anchor) == 0 or len(prop) == 0:
        print(f"Dados insuficientes para {img_name}")
        return
        
    plt.figure(figsize=(8, 6))
    
    # Plot Anchor
    plt.plot(anchor['bpp'], anchor['psnr'], marker='o', linestyle='-', color='red', label='Image-GS Original (Âncora)')
    
    # Plot Proposed
    plt.plot(prop['bpp'], prop['psnr'], marker='s', linestyle='-', color='blue', label='GaussianARM (Proposto)')
    
    plt.title(title)
    plt.xlabel("BPP (Bits Per Pixel) - Menor é melhor")
    plt.ylabel("PSNR (dB) - Maior é melhor")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save directly to the TCC figures folder
    save_path = os.path.join("tcc_diego", "figuras", filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Gerado: {save_path}")

# Plot Anime (Vitória)
plot_image('anime-2_2k', 'curva_anime.png', 'Curva Taxa-Distorção - Categoria: Anime (Sucesso)')

# Plot Photo 5 (Vitória Expressiva)
plot_image('photo-5_2k', 'curva_photo.png', 'Curva Taxa-Distorção - Categoria: Fotografia (Sucesso)')

# Plot Art (Derrota)
plot_image('art-1_2k', 'curva_art.png', 'Curva Taxa-Distorção - Categoria: Arte (Falha)')
