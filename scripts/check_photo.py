import pandas as pd
import numpy as np

df = pd.read_csv("resultados_experimentos.csv")
df_img = df[df['image_name'] == 'photo-1_2k']
anchor = df_img[df_img['lambda_rdo'] == 0.0].sort_values(by='bpp')
prop = df_img[df_img['lambda_rdo'] == 0.0005].sort_values(by='bpp')

print("Anchor:")
print(anchor[['num_gaussians', 'psnr', 'bpp']])
print("\nProposed:")
print(prop[['num_gaussians', 'psnr', 'bpp']])
