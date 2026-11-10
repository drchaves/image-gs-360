import pandas as pd
df = pd.read_csv("resultados_experimentos.csv")

anchor = df[(df['lambda_rdo'] == 0.0) & (df['num_gaussians'] == 100000)]
prop = df[(df['lambda_rdo'] == 0.0005) & (df['num_gaussians'] == 100000)]

print("Anchor (100k, L=0.0):")
print(f"MS-SSIM: {anchor['msssim'].mean():.4f}")
print(f"PSNR: {anchor['psnr'].mean():.2f}")
print(f"LPIPS: {anchor['lpips'].mean():.4f}")
print(f"BPP: {anchor['bpp'].mean():.4f}")

print("\nProposed (100k, L=0.0005):")
print(f"MS-SSIM: {prop['msssim'].mean():.4f}")
print(f"PSNR: {prop['psnr'].mean():.2f}")
print(f"LPIPS: {prop['lpips'].mean():.4f}")
print(f"BPP: {prop['bpp'].mean():.4f}")
