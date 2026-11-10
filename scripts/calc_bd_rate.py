import pandas as pd
import numpy as np

def bjontegaard_rate(rate1, psnr1, rate2, psnr2):
    """
    Calcula o BD-Rate (Bjontegaard Delta Rate) entre duas curvas de Taxa-Distorção.
    rate1, psnr1: pontos da curva âncora (Baseline)
    rate2, psnr2: pontos da curva proposta
    Retorna o BD-Rate em porcentagem (%). Valores negativos indicam ganho (compressão melhor).
    """
    log_rate1 = np.log10(rate1)
    log_rate2 = np.log10(rate2)
    
    # Ajuste polinomial de 3º grau (log_rate em função do PSNR)
    try:
        p1 = np.polyfit(psnr1, log_rate1, 3)
        p2 = np.polyfit(psnr2, log_rate2, 3)
    except:
        return float('nan')
    
    # Define o intervalo de integração (interseção dos PSNRs)
    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    
    if max_psnr <= min_psnr:
        print("Aviso: Curvas não se sobrepõem no eixo PSNR.")
        return float('nan')
    
    # Função auxiliar para integrar o polinômio
    def int_poly(p, x):
        return (p[0]/4) * x**4 + (p[1]/3) * x**3 + (p[2]/2) * x**2 + p[3] * x
        
    int1 = int_poly(p1, max_psnr) - int_poly(p1, min_psnr)
    int2 = int_poly(p2, max_psnr) - int_poly(p2, min_psnr)
    
    # Calcula a diferença média
    avg_diff = (int2 - int1) / (max_psnr - min_psnr)
    
    # Converte de log10 para porcentagem de variação de bitrate
    bd_rate = (10**avg_diff - 1) * 100
    return bd_rate

def main():
    csv_file = "resultados_experimentos.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Arquivo {csv_file} não encontrado. Aguarde o fim dos testes.")
        return

    images = df['image_name'].unique()
    
    bd_rates = []
    
    print("==================================================")
    print(" CÁLCULO DE BD-RATE (Bjontegaard Delta Rate)")
    print("==================================================")
    
    for img in images:
        df_img = df[df['image_name'] == img]
        
        # Curva Âncora: Lambda = 0.0, variando as gaussianas
        df_anchor = df_img[df_img['lambda_rdo'] == 0.0].sort_values(by='bpp')
        
        # Curva Proposta: Usando Lambda = 0.0005 e variando as gaussianas
        df_prop = df_img[df_img['lambda_rdo'] == 0.0005].sort_values(by='bpp')
        
        if len(df_anchor) < 4 or len(df_prop) < 4:
            print(f"[{img}] Pontos insuficientes para curva polinomial (precisa de pelo menos 4 pontos).")
            continue
            
        rate1 = df_anchor['bpp'].values
        psnr1 = df_anchor['psnr'].values
        
        rate2 = df_prop['bpp'].values
        psnr2 = df_prop['psnr'].values
        
        bd = bjontegaard_rate(rate1, psnr1, rate2, psnr2)
        bd_rates.append(bd)
        
        print(f"Imagem: {img}")
        print(f"  -> BD-Rate: {bd:+.2f} %")
        print("--------------------------------------------------")
        
    if bd_rates:
        mean_bd = np.nanmean(bd_rates)
        print(f"\nMÉDIA GLOBAL BD-RATE: {mean_bd:+.2f} %")
        print("==================================================")
        if mean_bd < 0:
            print("Conclusão: Seu método ECONOMIZA bits comparado à versão original! Sucesso!")
        else:
            print("Conclusão: Seu método consome mais bits. Reveja os hiperparâmetros.")

if __name__ == "__main__":
    main()
