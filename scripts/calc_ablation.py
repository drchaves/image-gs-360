import pandas as pd

df = pd.read_csv("resultados_experimentos.csv")

def print_ablation(num_gaussians):
    print(f"==================================================")
    print(f" ABLAÇÃO DO PARÂMETRO LAMBDA (Fixado: {num_gaussians} Gauss.)")
    print(f"==================================================")
    
    lambdas = [0.0, 0.0005, 0.001, 0.005, 0.01]
    
    for l in lambdas:
        df_l = df[(df['lambda_rdo'] == l) & (df['num_gaussians'] == num_gaussians)]
        
        if len(df_l) == 0:
            continue
            
        mean_psnr = df_l['psnr'].mean()
        mean_bpp = df_l['bpp'].mean()
        
        # Calcular a queda de PSNR e ganho de BPP em relação ao L=0.0
        anchor = df[(df['lambda_rdo'] == 0.0) & (df['num_gaussians'] == num_gaussians)]
        psnr_drop = anchor['psnr'].mean() - mean_psnr
        bpp_save = ((anchor['bpp'].mean() - mean_bpp) / anchor['bpp'].mean()) * 100
        
        if l == 0.0:
            print(f"Lambda = {l:<6} (Âncora): PSNR = {mean_psnr:.2f} dB | BPP = {mean_bpp:.4f} (Baseline)")
        else:
            print(f"Lambda = {l:<6} (RDO)   : PSNR = {mean_psnr:.2f} dB | BPP = {mean_bpp:.4f} "
                  f"[BPP: -{bpp_save:.1f}% | PSNR: -{psnr_drop:.2f} dB]")

print_ablation(100000)
