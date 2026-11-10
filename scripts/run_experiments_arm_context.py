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
# Configurações do Experimento: Ablação de Contexto do ARM (Arquitetura Shallow + Stabilize Grads)
# ==========================================
IMAGES_DIR = "media/images"
PREV_RESULTS_CSV = "resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv"
RESULTS_CSV = "resultados_diego_csvs/results_shallow_stab_context_4_8_16_32.csv"

MAX_STEPS = 50000
USE_STE = True
ARM_BITS = 8
ARM_ARCH = "shallow"
STABILIZE_GRADS = True

NUM_GAUSSIANS_LIST = [100000]
RATE_LOSS_RATIOS = [0.0005, 0.0025, 0.0075]
CONTEXT_SIZES = [4, 8, 16, 32]


def get_image_resolution(img_path):
    with Image.open(img_path) as img:
        return img.width, img.height


def import_context_16_from_prev_csv(fieldnames, completed_runs):
    """Importa os resultados já concluídos de context_size=16 (shallow + stabilize_grads=True) do CSV anterior."""
    if not os.path.isfile(PREV_RESULTS_CSV):
        return 0

    imported_rows = []
    with open(PREV_RESULTS_CSV, mode='r', newline='') as f_prev:
        reader = csv.DictReader(f_prev)
        for row in reader:
            arch = row.get('arm_arch', '').strip()
            stab = str(row.get('stabilize_grads', '')).strip().lower()
            if arch == ARM_ARCH and stab == str(STABILIZE_GRADS).lower():
                img_name = row.get('image_name', '').strip()
                num_g = str(row.get('num_gaussians', '')).strip()
                r_loss_str = f"{float(row.get('lambda_rdo', 0.0)):g}"
                key = (img_name, num_g, r_loss_str, arch, stab, "16")
                if key not in completed_runs:
                    new_row = {k: row.get(k, '') for k in fieldnames if k != 'arm_context'}
                    new_row['arm_context'] = 16
                    imported_rows.append(new_row)
                    completed_runs.add(key)

    if imported_rows:
        file_exists = os.path.isfile(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
        with open(RESULTS_CSV, mode='a', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(imported_rows)

    return len(imported_rows)


def main():
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    image_paths.sort()

    if not image_paths:
        print(f"Nenhuma imagem encontrada na pasta {IMAGES_DIR}/")
        return

    os.makedirs(os.path.dirname(RESULTS_CSV) or ".", exist_ok=True)
    fieldnames = [
        'image_name', 'num_gaussians', 'lambda_rdo', 'max_steps', 'final_step',
        'early_stopped', 'final_num_gaussians', 'use_ste', 'arm_bits',
        'arm_arch', 'arm_context', 'stabilize_grads', 'psnr', 'ws_psnr',
        'ssim', 'msssim', 'lpips', 'bytes', 'bpp_img', 'bpp_arm', 'bpp'
    ]

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
                    str(row.get('stabilize_grads', '')).strip().lower(),
                    str(row.get('arm_context', '16')).strip()
                )
                completed_runs.add(key)

    # Reaproveita automaticamente os 135 resultados (45 imagens x 3 lambdas) de C=16
    imported_count = import_context_16_from_prev_csv(fieldnames, completed_runs)
    if imported_count > 0:
        print(f"[Reaproveitamento] {imported_count} resultados de context_size=16 importados de {PREV_RESULTS_CSV} para {RESULTS_CSV}!")
        file_exists = True

    with open(RESULTS_CSV, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for img_path in image_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            w, h = get_image_resolution(img_path)
            num_pixels = w * h

            for ctx in CONTEXT_SIZES:
                for num_g in NUM_GAUSSIANS_LIST:
                    for r_loss in RATE_LOSS_RATIOS:
                        run_key = (
                            img_name,
                            str(num_g),
                            f"{float(r_loss):g}",
                            ARM_ARCH,
                            str(STABILIZE_GRADS).lower(),
                            str(ctx)
                        )
                        if run_key in completed_runs:
                            print(f"[{img_name}] Pulando (já concluído no CSV): {ARM_ARCH} | C={ctx} | Stabilize: {STABILIZE_GRADS} | {num_g} gauss | Lambda: {r_loss}")
                            continue

                        stab_tag = "_stab" if STABILIZE_GRADS else ""
                        exp_name = f"exp_{img_name}_{ARM_ARCH}{stab_tag}_c{ctx}_g{num_g}_L{r_loss}"
                        print(f"\n[{img_name}] Rodando: Arquitetura {ARM_ARCH} | Contexto C={ctx} | Stabilize: {STABILIZE_GRADS} | {num_g} gauss | Lambda: {r_loss}")

                        cmd = [
                            sys.executable, "main.py",
                            "--data_root", ".",
                            "--input_path", img_path,
                            "--exp_name", exp_name,
                            "--num_gaussians", str(num_g),
                            "--rate_loss_ratio", str(r_loss),
                            "--quantize",
                            "--max_steps", str(MAX_STEPS),
                            "--arm_arch", ARM_ARCH,
                            "--arm_context", str(ctx)
                        ]
                        if USE_STE:
                            cmd.extend(["--use_ste", "--arm_bits", str(ARM_BITS)])
                        if STABILIZE_GRADS:
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
                            'arm_arch': ARM_ARCH,
                            'arm_context': ctx,
                            'stabilize_grads': str(STABILIZE_GRADS),
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

