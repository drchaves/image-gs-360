import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bjontegaard_rate(rate1, psnr1, rate2, psnr2):
    log_rate1 = np.log10(rate1)
    log_rate2 = np.log10(rate2)
    try:
        p1 = np.polyfit(psnr1, log_rate1, 3)
        p2 = np.polyfit(psnr2, log_rate2, 3)
    except:
        return float('nan')
    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    if max_psnr <= min_psnr:
        return float('nan')
    def int_poly(p, x):
        return (p[0]/4) * x**4 + (p[1]/3) * x**3 + (p[2]/2) * x**2 + p[3] * x
    int1 = int_poly(p1, max_psnr) - int_poly(p1, min_psnr)
    int2 = int_poly(p2, max_psnr) - int_poly(p2, min_psnr)
    avg_diff = (int2 - int1) / (max_psnr - min_psnr)
    return (10**avg_diff - 1) * 100

def main():
    csv_file = "resultados_experimentos.csv"
    out_dir = "/home/diego/.gemini/antigravity-ide/brain/91f290a6-78d1-4708-85ef-a17ebba7b0e6"
    df = pd.read_csv(csv_file)
    
    images = df['image_name'].unique()
    bd_rates = []
    metrics = []
    
    # Process each image
    for img in images:
        df_img = df[df['image_name'] == img]
        
        # Anchor (lambda = 0)
        df_anchor = df_img[df_img['lambda_rdo'] == 0.0].sort_values(by='bpp')
        # Proposed (lambda = 0.0005)
        df_prop = df_img[df_img['lambda_rdo'] == 0.0005].sort_values(by='bpp')
        
        if len(df_anchor) < 4 or len(df_prop) < 4:
            continue
            
        rate1 = df_anchor['bpp'].values
        psnr1 = df_anchor['psnr'].values
        rate2 = df_prop['bpp'].values
        psnr2 = df_prop['psnr'].values
        
        bd = bjontegaard_rate(rate1, psnr1, rate2, psnr2)
        bd_rates.append(bd)
        metrics.append({'image': img, 'bd_rate': bd})
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(rate1, psnr1, marker='o', linestyle='-', color='red', label='Âncora (sem ARM)')
        plt.plot(rate2, psnr2, marker='s', linestyle='-', color='blue', label='Proposto (com GaussianARM)')
        plt.title(f"Curva RD - {img}\nBD-Rate: {bd:+.2f}%")
        plt.xlabel("BPP (Bits Per Pixel)")
        plt.ylabel("PSNR (dB)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plot_path = os.path.join(out_dir, f"plot_{img}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    
    # Calculate global average
    if bd_rates:
        avg_bd = np.mean(bd_rates)
    else:
        avg_bd = float('nan')
        
    print("MÉTRICAS GERADAS:")
    for m in metrics:
        print(f"{m['image']}: {m['bd_rate']:.2f}%")
    print(f"MÉDIA GLOBAL: {avg_bd:.2f}%")

if __name__ == '__main__':
    main()
