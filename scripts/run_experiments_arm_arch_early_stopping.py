import os
import sys
import glob
import subprocess

# Garante que binários do mesmo ambiente conda (como ninja e nvcc) estejam no PATH
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
import torch
import csv
from PIL import Image

# ==========================================
# Configurações do Experimento: Ablação de Arquiteturas do ARM
# ==========================================
IMAGES_DIR = "media/images"
RESULTS_CSV = "resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv"

MAX_STEPS = 50000
USE_STE = True
ARM_BITS = 8
STABILIZE_GRADS = [True, False]

NUM_GAUSSIANS_LIST = [100000]
#RATE_LOSS_RATIOS = [0.0001, 0.0005, 0.0010]
RATE_LOSS_RATIOS = [0.0005, 0.0025, 0.0075]
ARCHITECTURES = ["original", "shallow", "deep"]
#ARCHITECTURES = ["original"]


def get_image_resolution(img_path):
    with Image.open(img_path) as img:
        return img.width, img.height

def main():
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    image_paths.sort()
    
    if not image_paths:
        print(f"Nenhuma imagem encontrada na pasta {IMAGES_DIR}/")
        return

    os.makedirs(os.path.dirname(RESULTS_CSV) or ".", exist_ok=True)
    completed_runs = set()
    file_exists = os.path.isfile(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    if file_exists:
        with open(RESULTS_CSV, mode='r', newline='') as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                key = (
                    row.get('image_name', '').strip(),
                    str(row.get('num_gaussians', '')).strip(),
                    f"{float(row.get('lambda_rdo', 0.0)):g}",
                    row.get('arm_arch', '').strip(),
                    str(row.get('stabilize_grads', '')).strip().lower()
                )
                completed_runs.add(key)

    with open(RESULTS_CSV, mode='a', newline='') as csv_file:
        fieldnames = ['image_name', 'num_gaussians', 'lambda_rdo', 'max_steps', 'final_step', 'early_stopped', 'final_num_gaussians', 'use_ste', 'arm_bits', 'arm_arch', 'stabilize_grads', 'psnr', 'ws_psnr', 'ssim', 'msssim', 'lpips', 'bytes', 'bpp_img', 'bpp_arm', 'bpp']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for img_path in image_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            w, h = get_image_resolution(img_path)
            num_pixels = w * h
            for grad in STABILIZE_GRADS:
                for arch in ARCHITECTURES:
                    for num_g in NUM_GAUSSIANS_LIST:
                        for r_loss in RATE_LOSS_RATIOS:
                            run_key = (img_name, str(num_g), f"{float(r_loss):g}", arch, str(grad).lower())
                            if run_key in completed_runs:
                                print(f"[{img_name}] Pulando (já concluído no CSV): {arch} | Stabilize: {grad} | {num_g} gauss | Lambda: {r_loss}")
                                continue

                            stab_tag = "_stab" if grad else ""
                            exp_name = f"exp_{img_name}_{arch}{stab_tag}_g{num_g}_L{r_loss}"
                            print(f"\n[{img_name}] Rodando: Arquitetura {arch} | Stabilize: {grad} | {num_g} gauss | Lambda: {r_loss}")
                            
                            cmd = [
                                sys.executable, "main.py",
                                "--data_root", ".",
                                "--input_path", img_path,
                                "--exp_name", exp_name,
                                "--num_gaussians", str(num_g),
                                "--rate_loss_ratio", str(r_loss),
                                "--quantize",
                                "--max_steps", str(MAX_STEPS),
                                # "--early_stopping",
                                "--arm_arch", arch
                            ]
                            if USE_STE:
                                cmd.extend(["--use_ste", "--arm_bits", str(ARM_BITS)])
                            if grad:
                                cmd.append("--stabilize_grads")
                            
                            try:
                                subprocess.run(cmd, check=True)
                            except subprocess.CalledProcessError as e:
                                print(f"Erro ao rodar {exp_name}: {e}")
                                continue
                            
                            ckpt_search_path = f"results/{exp_name}/**/*.pt"
                            ckpt_files = glob.glob(ckpt_search_path, recursive=True)
                            
                            if not ckpt_files:
                                print(f"Atenção: Checkpoint não encontrado para {exp_name}")
                                continue
                                
                            ckpt_file = max(ckpt_files, key=os.path.getmtime)
                            ckpt_data = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                            
                            psnr = ckpt_data.get('psnr', 0.0)
                            ws_psnr = ckpt_data.get('ws_psnr', psnr)
                            ssim = ckpt_data.get('ssim', 0.0)
                            msssim = ckpt_data.get('msssim', 0.0)
                            lpips_val = ckpt_data.get('lpips', 0.0)
                            num_bytes = ckpt_data.get('bytes', 0.0)
                            num_bytes_img = ckpt_data.get('bytes_img', num_bytes)
                            num_bytes_arm = ckpt_data.get('bytes_arm', 0.0)
                            
                            bpp = (num_bytes * 8) / num_pixels
                            bpp_img = (num_bytes_img * 8) / num_pixels
                            bpp_arm = (num_bytes_arm * 8) / num_pixels
                            
                            final_step = ckpt_data.get('step', MAX_STEPS)
                            early_stopped = "True" if final_step < MAX_STEPS else "False"
                            final_num_gaussians = ckpt_data.get('num_gaussians_final', num_g)
                            
                            writer.writerow({
                                'image_name': img_name,
                                'num_gaussians': num_g,
                                'lambda_rdo': r_loss,
                                'max_steps': MAX_STEPS,
                                'final_step': final_step,
                                'early_stopped': early_stopped,
                                'final_num_gaussians': final_num_gaussians,
                                'use_ste': str(USE_STE),
                                'arm_bits': ARM_BITS if USE_STE else "-",
                                'arm_arch': arch,
                                'stabilize_grads': str(grad),
                                'psnr': f"{psnr:.2f}",
                                'ws_psnr': f"{ws_psnr:.2f}",
                                'ssim': f"{ssim:.4f}",
                                'msssim': f"{msssim:.4f}",
                                'lpips': f"{lpips_val:.4f}",
                                'bytes': f"{num_bytes:.1f}",
                                'bpp_img': f"{bpp_img:.4f}",
                                'bpp_arm': f"{bpp_arm:.4f}",
                                'bpp': f"{bpp:.4f}"
                            })
                            csv_file.flush()
                            completed_runs.add(run_key)
                            
                            print(f"-> Concluído! PSNR: {psnr:.2f} | WS-PSNR: {ws_psnr:.2f} | BPP: {bpp:.4f} | LPIPS: {lpips_val:.4f}")

if __name__ == "__main__":
    main()
