import pandas as pd

df = pd.read_csv("resultados_experimentos.csv")

def print_avg(num_gaussians):
    anchor = df[(df['lambda_rdo'] == 0.0) & (df['num_gaussians'] == num_gaussians)]
    prop = df[(df['lambda_rdo'] == 0.0005) & (df['num_gaussians'] == num_gaussians)]
    
    print(f"\n--- {num_gaussians} GAUSSIANAS (Média Global) ---")
    print("Âncora (Image-GS Original):")
    print(f"  PSNR: {anchor['psnr'].mean():.2f} dB")
    print(f"  BPP:  {anchor['bpp'].mean():.4f}")
    
    print("\nProposto (GaussianARM RDO):")
    print(f"  PSNR: {prop['psnr'].mean():.2f} dB")
    print(f"  BPP:  {prop['bpp'].mean():.4f}")
    
    economia_bpp = ((anchor['bpp'].mean() - prop['bpp'].mean()) / anchor['bpp'].mean()) * 100
    print(f"\nEconomia de arquivo: {economia_bpp:.2f}% menor!")

print_avg(10000)
print_avg(30000)
print_avg(50000)
print_avg(100000)
