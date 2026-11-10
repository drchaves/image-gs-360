import csv

old_rows = {}
with open('resultados_experimentos_antes_bpp_arm.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['image_name'], row['num_gaussians'], row['lambda_rdo'])
        old_rows[key] = row

new_rows = {}
with open('resultados_experimentos.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['image_name'], row['num_gaussians'], row['lambda_rdo'])
        new_rows[key] = row

from collections import defaultdict
splat_gains = defaultdict(list)
total_gains = defaultdict(list)
psnr_diffs = defaultdict(list)
bpp_arms = []

for key in old_rows:
    if key in new_rows and float(key[2]) > 0:
        old = old_rows[key]
        new = new_rows[key]
        
        num_g = int(key[1])
        bpp_old = float(old['bpp'])
        bpp_img = float(new['bpp_img'])
        bpp_new = float(new['bpp'])
        psnr_old = float(old['psnr'])
        psnr_new = float(new['psnr'])
        
        gain_splat = (bpp_old - bpp_img) / bpp_old * 100
        gain_total = (bpp_old - bpp_new) / bpp_old * 100
        psnr_diff = psnr_new - psnr_old
        
        splat_gains[num_g].append(gain_splat)
        total_gains[num_g].append(gain_total)
        psnr_diffs[num_g].append(psnr_diff)
        bpp_arms.append(float(new['bpp_arm']))

print("=== Média de Ganho nas Features (bpp_img vs bpp_old) por Quantidade de Gaussianas ===")
for k in sorted(splat_gains.keys()):
    print(f"{k}: {sum(splat_gains[k])/len(splat_gains[k]):.2f}%")

print("\n=== Média de Ganho TOTAL (bpp_new vs bpp_old) por Quantidade de Gaussianas ===")
for k in sorted(total_gains.keys()):
    print(f"{k}: {sum(total_gains[k])/len(total_gains[k]):.2f}%")

print("\n=== Diferença Média de PSNR ===")
for k in sorted(psnr_diffs.keys()):
    print(f"{k}: {sum(psnr_diffs[k])/len(psnr_diffs[k]):.4f} dB")

print(f"\nBPP ARM médio: {sum(bpp_arms)/len(bpp_arms):.4f}")

