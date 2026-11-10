import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    csv_file = 'resultados_arm_arch_50k.csv'
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # Pegando as imagens únicas
    images = df['image_name'].unique()
    
    out_dir = 'plots_arm_arch'
    os.makedirs(out_dir, exist_ok=True)
    
    # Define as cores/estilos para cada arquitetura
    sns.set_theme(style="whitegrid")
    
    for img in images:
        df_img = df[df['image_name'] == img]
        
        plt.figure(figsize=(8, 6))
        
        for arch in df['arm_arch'].unique():
            df_arch = df_img[df_img['arm_arch'] == arch].sort_values(by='bpp')
            
            plt.plot(df_arch['bpp'], df_arch['psnr'], marker='o', label=f'Arch: {arch}')
            
        plt.title(f'Rate-Distortion (PSNR vs BPP) - {img}')
        plt.xlabel('Bits per Pixel (BPP)')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.grid(True)
        
        out_path = os.path.join(out_dir, f'rd_curve_{img}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Plots salvos na pasta {out_dir}/")

    # Também vamos criar um gráfico médio de todas as imagens, para ver o comportamento geral
    # Como as imagens tem PSNRs diferentes, podemos normalizar ou apenas mostrar a média
    # O jeito mais comum é calcular a média de BPP e PSNR para cada (arch, lambda_rdo)
    df_mean = df.groupby(['arm_arch', 'lambda_rdo']).mean(numeric_only=True).reset_index()
    
    plt.figure(figsize=(8, 6))
    for arch in df_mean['arm_arch'].unique():
        df_arch = df_mean[df_mean['arm_arch'] == arch].sort_values(by='bpp')
        plt.plot(df_arch['bpp'], df_arch['psnr'], marker='s', linewidth=2, markersize=8, label=f'Arch: {arch} (Média)')
        
    plt.title('Rate-Distortion Médio (Todas as imagens)')
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(out_dir, 'rd_curve_media_geral.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot médio salvo em {out_dir}/rd_curve_media_geral.png")

if __name__ == '__main__':
    main()

