import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file = "resultados_arm_arch_early_stopping_fieldnames.csv"
df = pd.read_csv(csv_file)
images = df['image_name'].unique()

out_dir = "/home/diego/.gemini/antigravity-ide/brain/91f290a6-78d1-4708-85ef-a17ebba7b0e6"

for img in images:
    df_img = df[df['image_name'] == img].sort_values(by='bpp')
    plt.figure(figsize=(8,6))
    plt.plot(df_img['bpp'], df_img['psnr'], marker='o', label='Early Stopping (max 10k steps)')
    plt.xlabel('BPP')
    plt.ylabel('PSNR (dB)')
    plt.title(f'RD Curve - {img} (Varying lambda)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f'rd_curve_es_{img}.png'))
    plt.close()
