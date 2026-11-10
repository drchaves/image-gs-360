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
    csv_old = "resultados_arm_arch_50k.csv"
    csv_new = "resultados_arm_arch_early_stopping_fieldnames.csv"
    out_dir = "/home/diego/.gemini/antigravity-ide/brain/91f290a6-78d1-4708-85ef-a17ebba7b0e6"
    
    df_old = pd.read_csv(csv_old)
    df_new = pd.read_csv(csv_new)
    
    # Filter for 'original' architecture
    df_old = df_old[df_old['arm_arch'] == 'original']
    df_new = df_new[df_new['arm_arch'] == 'original']
    
    images = df_old['image_name'].unique()
    
    report_lines = ["# Análise de Ablação: Early Stopping vs Sem Early Stopping\n"]
    report_lines.append("| Imagem | BD-Rate (BPP) | Tempo/Passos |\n")
    report_lines.append("|--------|---------------|--------------|\n")
    
    bd_rates = []
    
    for img in images:
        df_old_img = df_old[df_old['image_name'] == img].sort_values(by='bpp')
        df_new_img = df_new[df_new['image_name'] == img].sort_values(by='bpp')
        
        if len(df_old_img) < 3 or len(df_new_img) < 3:
            continue
            
        rate1 = df_old_img['bpp'].values
        psnr1 = df_old_img['psnr'].values
        rate2 = df_new_img['bpp'].values
        psnr2 = df_new_img['psnr'].values
        
        bd_rate = bjontegaard_rate(rate1, psnr1, rate2, psnr2)
        bd_rates.append(bd_rate)
        
        # Obter os passos finais
        steps_old = df_old_img['max_steps'].iloc[0]
        steps_new = df_new_img['final_step'].iloc[0] if 'final_step' in df_new_img.columns else "N/A"
        
        report_lines.append(f"| {img} | {bd_rate:.2f}% | 50.000 vs {steps_new} |")
        
        # Plot
        plt.figure(figsize=(8,6))
        plt.plot(rate1, psnr1, marker='o', label=f'Sem Early Stopping (50k passos)')
        plt.plot(rate2, psnr2, marker='^', label=f'Com Early Stopping (~{steps_new} passos)')
        plt.xlabel('BPP (Bits Per Pixel)')
        plt.ylabel('PSNR (dB)')
        plt.title(f'Early Stopping Ablation - {img}')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(out_dir, f'plot_es_ablation_{img}.png')
        plt.savefig(plot_path)
        plt.close()
        
    avg_bd = np.nanmean(bd_rates)
    report_lines.append(f"\n**Média de BD-Rate:** {avg_bd:.2f}%\n")
    report_lines.append("\n> **Nota:** Valores negativos indicam que o Early Stopping *economizou* bits mantendo a mesma qualidade. Valores positivos baixos indicam que houve uma perda mínima de qualidade, o que muitas vezes compensa a enorme economia de processamento (de 50 mil passos para ~4 mil).")
    
    with open(os.path.join(out_dir, "relatorio_early_stopping.md"), "w") as f:
        f.write("\n".join(report_lines))
        
if __name__ == "__main__":
    main()
