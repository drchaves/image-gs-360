import pandas as pd
import numpy as np

def bjontegaard_rate(rate1, psnr1, rate2, psnr2):
    log_rate1 = np.log10(rate1)
    log_rate2 = np.log10(rate2)
    p1 = np.polyfit(psnr1, log_rate1, 2) # Use degree 2 for 3 points
    p2 = np.polyfit(psnr2, log_rate2, 2)
    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    def int_poly(p, x):
        return (p[0]/3) * x**3 + (p[1]/2) * x**2 + p[2] * x
    int1 = int_poly(p1, max_psnr) - int_poly(p1, min_psnr)
    int2 = int_poly(p2, max_psnr) - int_poly(p2, min_psnr)
    avg_diff = (int2 - int1) / (max_psnr - min_psnr)
    return (10**avg_diff - 1) * 100

df = pd.read_csv("resultados_experimentos.csv")
df_img = df[df['image_name'] == 'photo-1_2k']
anchor = df_img[df_img['lambda_rdo'] == 0.0].sort_values(by='bpp')
prop = df_img[df_img['lambda_rdo'] == 0.0005].sort_values(by='bpp')

# Filter out the 100k gaussian point
anchor = anchor[anchor['num_gaussians'] < 100000]
prop = prop[prop['num_gaussians'] < 100000]

rate1 = anchor['bpp'].values
psnr1 = anchor['psnr'].values
rate2 = prop['bpp'].values
psnr2 = prop['psnr'].values

print("BD-Rate sem 100k:", bjontegaard_rate(rate1, psnr1, rate2, psnr2))
