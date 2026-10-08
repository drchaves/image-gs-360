import os
import glob
import subprocess
import torch
import csv
from PIL import Image

# ==========================================
# Configurações do Experimento
# ==========================================
IMAGES_DIR = "media/images"
RESULTS_CSV = "resultados_experimentos.csv"

# Grade de hiperparâmetros para varredura
NUM_GAUSSIANS_LIST = [10000, 30000, 50000, 100000]
RATE_LOSS_RATIOS = [0.0, 0.0005, 0.001, 0.005, 0.01]

def get_image_resolution(img_path):
    with Image.open(img_path) as img:
        return img.width, img.height

def main():
    # Encontra todas as imagens (png, jpg, jpeg) na pasta
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    
    if not image_paths:
        print(f"Nenhuma imagem encontrada na pasta {IMAGES_DIR}/")
        return

    # Prepara o CSV
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, mode='a', newline='') as csv_file:
        fieldnames = ['image_name', 'num_gaussians', 'lambda_rdo', 'psnr', 'ssim', 'msssim', 'lpips', 'bytes', 'bpp_img', 'bpp_arm', 'bpp']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for img_path in image_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            w, h = get_image_resolution(img_path)
            num_pixels = w * h

            for num_g in NUM_GAUSSIANS_LIST:
                for r_loss in RATE_LOSS_RATIOS:
                    exp_name = f"exp_{img_name}_g{num_g}_L{r_loss}"
                    print(f"\n[{img_name}] Rodando: {num_g} gaussianas | Lambda: {r_loss}")
                    
                    # Monta o comando de execução
                    cmd = [
                        "python", "main.py",
                        "--data_root", ".",  # <-- Evita que o main.py coloque 'media/' na frente do caminho
                        "--input_path", img_path,
                        "--exp_name", exp_name,
                        "--num_gaussians", str(num_g),
                        "--rate_loss_ratio", str(r_loss),
                        "--quantize",
                        "--max_steps", "10000", # Ajuste o número de passos conforme necessário
                        # A flag --use_erp foi omitida pois focaremos em 2D planar
                    ]
                    
                    # Executa o main.py
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Erro ao rodar {exp_name}: {e}")
                        continue
                    
                    # Procura o arquivo .pt gerado na pasta de resultados
                    # O main.py salva em: results/<exp_name>/<gaussian_cfg_loss_cfg>/ckpt_step-XXXX.pt
                    ckpt_search_path = f"results/{exp_name}/**/*.pt"
                    ckpt_files = glob.glob(ckpt_search_path, recursive=True)
                    
                    if not ckpt_files:
                        print(f"Atenção: Checkpoint não encontrado para {exp_name}")
                        continue
                        
                    # Pega o último checkpoint gerado
                    ckpt_file = sorted(ckpt_files)[-1]
                    
                    # Lê as métricas usando o decodificador simulado salvo no .pt
                    ckpt_data = torch.load(ckpt_file, map_location='cpu')
                    
                    psnr = ckpt_data.get('psnr', 0.0)
                    ssim = ckpt_data.get('ssim', 0.0)
                    msssim = ckpt_data.get('msssim', 0.0)
                    lpips_val = ckpt_data.get('lpips', 0.0)
                    num_bytes = ckpt_data.get('bytes', 0.0)
                    num_bytes_img = ckpt_data.get('bytes_img', num_bytes)
                    num_bytes_arm = ckpt_data.get('bytes_arm', 0.0)
                    
                    # Calcula o bpp real da imagem
                    bpp = (num_bytes * 8) / num_pixels
                    bpp_img = (num_bytes_img * 8) / num_pixels
                    bpp_arm = (num_bytes_arm * 8) / num_pixels
                    
                    # Salva no CSV imediatamente
                    writer.writerow({
                        'image_name': img_name,
                        'num_gaussians': num_g,
                        'lambda_rdo': r_loss,
                        'psnr': f"{psnr:.2f}",
                        'ssim': f"{ssim:.4f}",
                        'msssim': f"{msssim:.4f}",
                        'lpips': f"{lpips_val:.4f}",
                        'bytes': f"{num_bytes:.1f}",
                        'bpp_img': f"{bpp_img:.4f}",
                        'bpp_arm': f"{bpp_arm:.4f}",
                        'bpp': f"{bpp:.4f}"
                    })
                    csv_file.flush() # Força a escrita no disco
                    
                    print(f"-> Concluído! PSNR: {psnr:.2f} | BPP: {bpp:.4f} | LPIPS: {lpips_val:.4f}")

if __name__ == "__main__":
    main()
