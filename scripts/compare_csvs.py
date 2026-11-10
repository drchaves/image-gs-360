import pandas as pd
import numpy as np
import sys

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

def get_bd_rates(csv_file):
    df = pd.read_csv(csv_file)
    images = df['image_name'].unique()
    results = {}
    for img in images:
        df_img = df[df['image_name'] == img]
        df_anchor = df_img[df_img['lambda_rdo'] == 0.0].sort_values(by='bpp')
        df_prop = df_img[df_img['lambda_rdo'] == 0.0005].sort_values(by='bpp')
        if len(df_anchor) < 4 or len(df_prop) < 4:
            continue
        rate1 = df_anchor['bpp'].values
        psnr1 = df_anchor['psnr'].values
        rate2 = df_prop['bpp'].values
        psnr2 = df_prop['psnr'].values
        bd = bjontegaard_rate(rate1, psnr1, rate2, psnr2)
        results[img] = bd
    return results

def main():
    csv_antes = "resultados_experimentos_antes_bpp_arm.csv"
    csv_depois = "resultados_experimentos.csv"
    
    bd_antes = get_bd_rates(csv_antes)
    bd_depois = get_bd_rates(csv_depois)
    
    print(f"{'Imagem':<20} | {'Antes (sem ARM)':<15} | {'Depois (com ARM)':<15} | {'Diferença'}")
    print("-" * 70)
    for img in bd_antes:
        v_antes = bd_antes[img]
        v_depois = bd_depois.get(img, float('nan'))
        diff = v_depois - v_antes
        print(f"{img:<20} | {v_antes:>14.2f}% | {v_depois:>14.2f}% | {diff:>+8.2f}%")

if __name__ == '__main__':
    main()
