import pandas as pd
import numpy as np

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

df = pd.read_csv("resultados_experimentos.csv")
images = df['image_name'].unique()
for img in images:
    df_img = df[df['image_name'] == img]
    df_anchor = df_img[df_img['lambda_rdo'] == 0.0].sort_values(by='bpp')
    df_prop = df_img[df_img['lambda_rdo'] == 0.0005].sort_values(by='bpp')
    bd = bjontegaard_rate(df_anchor['bpp'].values, df_anchor['psnr'].values, df_prop['bpp'].values, df_prop['psnr'].values)
    print(f"{img}: {bd:.2f}%")
