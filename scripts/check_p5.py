import pandas as pd
df = pd.read_csv("resultados_experimentos.csv")
p5 = df[df['image_name'] == 'photo-5_2k']
print("photo-5_2k:")
print(p5[['lambda_rdo', 'num_gaussians', 'psnr', 'bpp']].sort_values(['lambda_rdo', 'num_gaussians']))
