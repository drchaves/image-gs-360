import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def bjontegaard_rate(rate1, psnr1, rate2, psnr2):
    log_rate1 = np.log10(rate1)
    log_rate2 = np.log10(rate2)
    
    # Polinômio de 2º grau pois temos 3 pontos por curva
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

def get_curve_points(data, img, arch, stab, metric='ws_psnr'):
    pts = [
        r for r in data
        if r['image_name'] == img
        and r['arm_arch'] == arch
        and r['stabilize_grads'] == stab
    ]
    if len(pts) < 3:
        return None, None
    # Ordena por PSNR/WS-PSNR crescente (ou lambda decrescente) para o ajuste polinomial
    pts = sorted(pts, key=lambda x: float(x[metric]))
    rates = np.array([float(x['bpp']) for x in pts])
    psnrs = np.array([float(x[metric]) for x in pts])
    return rates, psnrs

def plot_curves(data, images, architectures, metric='ws_psnr', out_dir='plots_stabilize_grads'):
    os.makedirs(out_dir, exist_ok=True)
    metric_label = "WS-PSNR (dB)" if metric == 'ws_psnr' else "PSNR (dB)"
    
    colors = {'original': '#1f77b4', 'shallow': '#ff7f0e', 'deep': '#2ca02c'}
    markers = {'True': 'o', 'False': 's'}
    linestyles = {'True': '-', 'False': '--'}
    
    # 1. Gráfico médio geral (média de BPP e WS-PSNR para cada combinação de arch, stabilize_grads e lambda_rdo)
    lambdas = sorted(list(set(float(r['lambda_rdo']) for r in data)), reverse=True)
    
    plt.figure(figsize=(10, 6))
    for arch in architectures:
        for stab in ['False', 'True']:
            mean_bpps = []
            mean_psnrs = []
            for lmb in lambdas:
                rows = [
                    r for r in data
                    if r['arm_arch'] == arch
                    and r['stabilize_grads'] == stab
                    and abs(float(r['lambda_rdo']) - lmb) < 1e-9
                ]
                if rows:
                    mean_bpps.append(np.mean([float(r['bpp']) for r in rows]))
                    mean_psnrs.append(np.mean([float(r[metric]) for r in rows]))
            if mean_bpps:
                stab_label = "Com Stabilize" if stab == 'True' else "Sem Stabilize"
                plt.plot(
                    mean_bpps, mean_psnrs,
                    color=colors.get(arch, None),
                    linestyle=linestyles[stab],
                    marker=markers[stab],
                    linewidth=2,
                    markersize=7,
                    label=f"{arch.capitalize()} ({stab_label})"
                )
                
    plt.title(f"Curva Rate-Distortion Média ({len(images)} imagens) - {metric_label} vs BPP")
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel(metric_label)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    avg_plot_path = os.path.join(out_dir, f"rd_media_geral_{metric}.png")
    plt.savefig(avg_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Gráfico salvo] {avg_plot_path}")

    # 2. Gráfico médio separado por arquitetura (painel 1x3)
    fig, axes = plt.subplots(1, len(architectures), figsize=(5 * len(architectures), 4.5), sharey=True)
    if len(architectures) == 1:
        axes = [axes]
    for ax, arch in zip(axes, architectures):
        for stab in ['False', 'True']:
            mean_bpps = []
            mean_psnrs = []
            for lmb in lambdas:
                rows = [
                    r for r in data
                    if r['arm_arch'] == arch
                    and r['stabilize_grads'] == stab
                    and abs(float(r['lambda_rdo']) - lmb) < 1e-9
                ]
                if rows:
                    mean_bpps.append(np.mean([float(r['bpp']) for r in rows]))
                    mean_psnrs.append(np.mean([float(r[metric]) for r in rows]))
            if mean_bpps:
                stab_label = "Com Stabilize (True)" if stab == 'True' else "Sem Stabilize (False)"
                ax.plot(
                    mean_bpps, mean_psnrs,
                    linestyle=linestyles[stab],
                    marker=markers[stab],
                    linewidth=2,
                    markersize=7,
                    label=stab_label
                )
        ax.set_title(f"Arquitetura: {arch.capitalize()}")
        ax.set_xlabel("BPP")
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
    axes[0].set_ylabel(metric_label)
    fig.suptitle(f"Comparação Stabilize Grads por Arquitetura ({metric_label} Médio)", fontsize=13)
    plt.tight_layout()
    panel_plot_path = os.path.join(out_dir, f"rd_painel_arquiteturas_{metric}.png")
    plt.savefig(panel_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Gráfico salvo] {panel_plot_path}")

    # 3. Gráficos individuais por imagem
    indiv_dir = os.path.join(out_dir, "por_imagem")
    os.makedirs(indiv_dir, exist_ok=True)
    for img in images:
        plt.figure(figsize=(8, 5.5))
        for arch in architectures:
            for stab in ['False', 'True']:
                r_pts, p_pts = get_curve_points(data, img, arch, stab, metric=metric)
                if r_pts is not None:
                    stab_label = "Stab=True" if stab == 'True' else "Stab=False"
                    plt.plot(
                        r_pts, p_pts,
                        color=colors.get(arch, None),
                        linestyle=linestyles[stab],
                        marker=markers[stab],
                        label=f"{arch} ({stab_label})"
                    )
        plt.title(f"Rate-Distortion ({metric_label} vs BPP) - {img}")
        plt.xlabel("Bits Per Pixel (BPP)")
        plt.ylabel(metric_label)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=9)
        img_plot_path = os.path.join(indiv_dir, f"rd_{img}_{metric}.png")
        plt.savefig(img_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
    print(f"[Gráficos individuais salvos] {len(images)} imagens em {indiv_dir}/")


def main():
    csv_file = "resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv"
    if not os.path.exists(csv_file):
        alt_path = os.path.join(os.path.dirname(__file__), "..", csv_file)
        if os.path.exists(alt_path):
            csv_file = alt_path

    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    images = list(dict.fromkeys([r['image_name'] for r in data]))
    architectures = list(dict.fromkeys([r['arm_arch'] for r in data]))
    
    for metric in ['ws_psnr', 'psnr']:
        metric_title = "WS-PSNR" if metric == 'ws_psnr' else "PSNR"
        print("\n======================================================================")
        print(f" 1. BD-RATE ({metric_title}): Stabilize Grads = True vs False [Baseline]")
        print("======================================================================")
        
        for arch in architectures:
            print(f"\n>>> Arquitetura: {arch} <<<")
            bd_stab = []
            valid_count = 0
            for img in images:
                r_base, p_base = get_curve_points(data, img, arch, 'False', metric=metric)
                r_stab, p_stab = get_curve_points(data, img, arch, 'True', metric=metric)
                if r_base is None or r_stab is None:
                    print(f"  {img:<15}: Dados incompletos (r_base={r_base is not None}, r_stab={r_stab is not None})")
                    continue
                bd = bjontegaard_rate(r_base, p_base, r_stab, p_stab)
                if not np.isnan(bd):
                    bd_stab.append(bd)
                    valid_count += 1
                    if metric == 'ws_psnr':
                        print(f"  {img:<15}: {bd:+7.2f} %")
                else:
                    if metric == 'ws_psnr':
                        print(f"  {img:<15}:     NaN")
                    
            if bd_stab:
                mean_bd = np.nanmean(bd_stab)
                median_bd = np.nanmedian(bd_stab)
                # Média aparada (sem extremos > 100% causados por extrapolação de polyfit grau 2)
                filtered_bd = [x for x in bd_stab if abs(x) < 100]
                mean_filt = np.mean(filtered_bd) if filtered_bd else float('nan')
                print(f"  --> RESUMO {arch:<8} | Média: {mean_bd:+7.2f}% | Mediana: {median_bd:+7.2f}% | Média s/ outliers (<100%): {mean_filt:+7.2f}% ({len(filtered_bd)}/{valid_count} imgs)")
            else:
                print(f"  Arquitetura {arch:<10} | Sem curvas válidas.")

        print("\n======================================================================")
        print(f" 2. BD-RATE ({metric_title}): Arquiteturas (Shallow/Deep vs Original)")
        print("======================================================================")
        for stab in ['False', 'True']:
            stab_str = "Com Stabilize (True)" if stab == 'True' else "Sem Stabilize (False)"
            print(f"\n  --- {stab_str} ---")
            for arch in ['shallow', 'deep']:
                if arch not in architectures:
                    continue
                bd_arch = []
                valid_count = 0
                for img in images:
                    r_orig, p_orig = get_curve_points(data, img, 'original', stab, metric=metric)
                    r_comp, p_comp = get_curve_points(data, img, arch, stab, metric=metric)
                    if r_orig is None or r_comp is None:
                        continue
                    bd = bjontegaard_rate(r_orig, p_orig, r_comp, p_comp)
                    if not np.isnan(bd):
                        bd_arch.append(bd)
                        valid_count += 1
                if bd_arch:
                    mean_bd = np.nanmean(bd_arch)
                    median_bd = np.nanmedian(bd_arch)
                    print(f"  {arch.capitalize():<8} vs Original | Média: {mean_bd:+7.2f}% | Mediana: {median_bd:+7.2f}% | ({valid_count}/{len(images)} imagens válidas)")

    # Gera os gráficos WS-PSNR x BPP
    plot_curves(data, images, architectures, metric='ws_psnr', out_dir='plots_stabilize_grads')

if __name__ == "__main__":
    main()
