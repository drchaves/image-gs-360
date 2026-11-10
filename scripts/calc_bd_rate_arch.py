import csv
import os
import numpy as np

def bjontegaard_rate(rate1, psnr1, rate2, psnr2):
    log_rate1 = np.log10(rate1)
    log_rate2 = np.log10(rate2)
    
    # Polinômio de 2º grau pois temos 3 pontos por curva (lambdas: 0.0001, 0.0005, 0.001)
    try:
        p1 = np.polyfit(psnr1, log_rate1, 2)
        p2 = np.polyfit(psnr2, log_rate2, 2)
    except Exception:
        return float('nan')
    
    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    
    if max_psnr <= min_psnr:
        return float('nan')
        
    def int_poly(p, x):
        return (p[0]/3) * x**3 + (p[1]/2) * x**2 + p[2] * x
        
    int1 = int_poly(p1, max_psnr) - int_poly(p1, min_psnr)
    int2 = int_poly(p2, max_psnr) - int_poly(p2, min_psnr)
    
    avg_diff = (int2 - int1) / (max_psnr - min_psnr)
    return (10**avg_diff - 1) * 100


def get_curve(data, img, arch, stab):
    pts = [
        r for r in data
        if r['image_name'] == img
        and r['arm_arch'] == arch
        and r['stabilize_grads'].strip().lower() == str(stab).lower()
    ]
    if len(pts) < 3:
        return None, None
    # Ordena por PSNR para o ajuste de log(rate) em função de PSNR
    pts = sorted(pts, key=lambda x: float(x['psnr']))
    rate = np.array([float(x['bpp']) for x in pts])
    psnr = np.array([float(x['psnr']) for x in pts])
    return rate, psnr


def main():
    csv_file = "resultados_diego_csvs/resultados_arm_archs_stabilize_grads_lambda_espacado.csv"
    if not os.path.exists(csv_file) and os.path.exists(os.path.join("..", csv_file)):
        csv_file = os.path.join("..", csv_file)

    data = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Mantém a ordem de aparição das imagens no CSV
    images = list(dict.fromkeys(r['image_name'] for r in data))
    archs = ["original", "shallow", "deep"]
    stabs = [False, True]

    # =========================================================================
    # 1. Comparação de todas as curvas contra a Âncora: (original, stabilize_grads=False)
    # =========================================================================
    print("==========================================================================")
    print(" 1) BD-RATE vs. BASELINE (original | stabilize_grads=False)")
    print("    (Valores negativos indicam economia de bits / melhor compressão)")
    print("==========================================================================")

    configs = [
        ("original", True),
        ("shallow", False),
        ("shallow", True),
        ("deep", False),
        ("deep", True),
    ]
    bd_vs_anchor = {cfg: [] for cfg in configs}

    for img in images:
        print(f"\nImagem: {img}")
        r_base, p_base = get_curve(data, img, "original", False)
        if r_base is None:
            print("  [!] Curva baseline (original, False) incompleta.")
            continue

        for arch, stab in configs:
            r_cur, p_cur = get_curve(data, img, arch, stab)
            if r_cur is None:
                continue
            bd = bjontegaard_rate(r_base, p_base, r_cur, p_cur)
            if not np.isnan(bd):
                bd_vs_anchor[(arch, stab)].append(bd)
            label = f"{arch:8s} | stabilize_grads={str(stab):5s}"
            print(f"  -> BD-Rate ({label}): {bd:+.2f} %")

    print("\n--------------------------------------------------------------------------")
    print(" MÉDIA GLOBAL vs. (original | stabilize_grads=False):")
    for arch, stab in configs:
        vals = bd_vs_anchor[(arch, stab)]
        mean_bd = np.nanmean(vals) if vals else float('nan')
        label = f"{arch:8s} | stabilize_grads={str(stab):5s}"
        print(f"  * {label}: {mean_bd:+.2f} %  (válido em {len(vals)}/{len(images)} imagens)")

    # =========================================================================
    # 2. Impacto direto de --stabilize_grads dentro de cada arquitetura (True vs False)
    # =========================================================================
    print("\n==========================================================================")
    print(" 2) IMPACTO DIRETO DO --stabilize_grads POR ARQUITETURA (True vs. False)")
    print("==========================================================================")
    bd_stab_gain = {arch: [] for arch in archs}

    for img in images:
        print(f"\nImagem: {img}")
        for arch in archs:
            r_false, p_false = get_curve(data, img, arch, False)
            r_true, p_true = get_curve(data, img, arch, True)
            if r_false is None or r_true is None:
                continue
            bd = bjontegaard_rate(r_false, p_false, r_true, p_true)
            if not np.isnan(bd):
                bd_stab_gain[arch].append(bd)
            print(f"  -> {arch:8s} (True vs False): {bd:+.2f} %")

    print("\n--------------------------------------------------------------------------")
    print(" MÉDIA GLOBAL DO GANHO COM --stabilize_grads (True vs. False):")
    for arch in archs:
        vals = bd_stab_gain[arch]
        mean_bd = np.nanmean(vals) if vals else float('nan')
        print(f"  * {arch:8s}: {mean_bd:+.2f} %  (válido em {len(vals)}/{len(images)} imagens)")

    # =========================================================================
    # 3. Comparação entre arquiteturas COM estabilização ligada (vs. original | True)
    # =========================================================================
    print("\n==========================================================================")
    print(" 3) ABLAÇÃO DE ARQUITETURAS COM ESTABILIZAÇÃO (vs. original | True)")
    print("==========================================================================")
    bd_arch_stab = {"shallow": [], "deep": []}

    for img in images:
        r_orig_t, p_orig_t = get_curve(data, img, "original", True)
        if r_orig_t is None:
            continue
        print(f"\nImagem: {img}")
        for arch in ["shallow", "deep"]:
            r_cur, p_cur = get_curve(data, img, arch, True)
            if r_cur is None:
                continue
            bd = bjontegaard_rate(r_orig_t, p_orig_t, r_cur, p_cur)
            if not np.isnan(bd):
                bd_arch_stab[arch].append(bd)
            print(f"  -> {arch:8s} (True) vs original (True): {bd:+.2f} %")

    print("\n--------------------------------------------------------------------------")
    print(" MÉDIA GLOBAL ENTRE ARQUITETURAS ESTABILIZADAS (vs. original | True):")
    for arch in ["shallow", "deep"]:
        vals = bd_arch_stab[arch]
        mean_bd = np.nanmean(vals) if vals else float('nan')
        print(f"  * {arch:8s} (True): {mean_bd:+.2f} %  (válido em {len(vals)}/{len(images)} imagens)")
    print("==========================================================================")


if __name__ == "__main__":
    main()

