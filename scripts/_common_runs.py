"""Utilitários compartilhados pelos scripts de execução da fase final
(run_final_comparison.py e run_extras.py).

Padrão idêntico ao dos scripts anteriores: execute SEMPRE a partir da raiz do repositório
(o `main.py` e `media/images` são caminhos relativos).
"""
import csv
import glob
import os
import subprocess
import sys

# Garante que ninja/nvcc do ambiente conda estejam no PATH (necessário para compilar a extensão CUDA)
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import torch  # noqa: E402  (após ajustar o PATH)
from PIL import Image  # noqa: E402

IMAGES_DIR = "media/images"

FIELDNAMES = [
    'image_name', 'method', 'config', 'num_gaussians', 'lambda_rdo', 'seed',
    'arm_arch', 'arm_context', 'use_ste', 'arm_bits', 'stabilize_grads',
    'log_scale_min', 'log_scale_max', 'stab_disable',
    'max_steps', 'final_step', 'final_num_gaussians',
    'psnr', 'ssim', 'msssim', 'lpips', 'bytes', 'bpp_img', 'bpp_arm', 'bpp', 'time_s',
]


def list_images(subset=None):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    paths.sort()
    if subset:
        wanted = set(subset)
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in wanted]
    return paths


def num_pixels(img_path):
    with Image.open(img_path) as img:
        return img.width * img.height


def make_key(image_name, num_g, lam, seed, config):
    return (str(image_name).strip(), str(int(float(num_g))), f"{float(lam):g}", str(int(float(seed))), str(config).strip())


def load_done_keys(csv_path):
    keys = set()
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                keys.add(make_key(row['image_name'], row['num_gaussians'], row['lambda_rdo'],
                                  row.get('seed', 123) or 123, row.get('config', '')))
    return keys


def append_rows(csv_path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDNAMES})


def run_one(img_path, exp_name, num_g, rate_loss, max_steps, seed=123, arm=None, retries=1):
    """Executa um treinamento via main.py e devolve as métricas do último checkpoint (ou None se falhar).

    arm=None  -> Image-GS original (rate_loss_ratio deve ser 0; sem ARM).
    arm=dict  -> chaves: arch, context, use_ste(bool), bits, stabilize(bool),
                 log_scale_min, log_scale_max, disable(lista de componentes da estabilização).
    """
    cmd = [
        sys.executable, "main.py",
        "--data_root", ".",
        "--input_path", img_path,
        "--exp_name", exp_name,
        "--num_gaussians", str(num_g),
        "--rate_loss_ratio", str(rate_loss),
        "--quantize",
        "--max_steps", str(max_steps),
        "--seed", str(seed),
    ]
    if arm is not None:
        cmd += ["--arm_arch", arm['arch'], "--arm_context", str(arm['context'])]
        if arm.get('use_ste', True):
            cmd += ["--use_ste", "--arm_bits", str(arm.get('bits', 8))]
        if arm.get('stabilize', False):
            cmd.append("--stabilize_grads")
        if arm.get('log_scale_min') is not None:
            cmd += ["--log_scale_min", str(arm['log_scale_min'])]
        if arm.get('log_scale_max') is not None:
            cmd += ["--log_scale_max", str(arm['log_scale_max'])]
        if arm.get('disable'):
            cmd += ["--stab_disable", *arm['disable']]  # nargs='*': deve ser o último argumento

    ok = False
    for attempt in range(retries + 1):
        try:
            subprocess.run(cmd, check=True)
            ok = True
            break
        except subprocess.CalledProcessError as e:
            print(f"[AVISO] Falha em {exp_name} (tentativa {attempt + 1}/{retries + 1}): {e}")
    if not ok:
        return None

    ckpts = glob.glob(f"results/{exp_name}/**/*.pt", recursive=True)
    if not ckpts:
        print(f"[AVISO] Checkpoint não encontrado para {exp_name}")
        return None
    d = torch.load(max(ckpts, key=os.path.getmtime), map_location='cpu', weights_only=False)
    n_pix = num_pixels(img_path)
    nbytes = d.get('bytes', 0.0)
    nbytes_img = d.get('bytes_img', nbytes)
    nbytes_arm = d.get('bytes_arm', 0.0)
    return {
        'max_steps': max_steps,
        'final_step': d.get('step', max_steps),
        'final_num_gaussians': d.get('num_gaussians_final', num_g),
        'psnr': f"{d.get('psnr', 0.0):.4f}",
        'ssim': f"{d.get('ssim', 0.0):.4f}",
        'msssim': f"{d.get('msssim', 0.0):.4f}",
        'lpips': f"{d.get('lpips', 0.0):.4f}",
        'bytes': f"{nbytes:.1f}",
        'bpp_img': f"{nbytes_img * 8 / n_pix:.4f}",
        'bpp_arm': f"{nbytes_arm * 8 / n_pix:.4f}",
        'bpp': f"{nbytes * 8 / n_pix:.4f}",
        'time_s': f"{d.get('time', 0.0):.1f}",
    }


def import_from_csv(src_csv, dst_csv, done_keys, selector, transform):
    """Importa linhas já existentes de outro CSV (reaproveitamento) sem rodar de novo.

    selector(row)->bool escolhe as linhas; transform(row)->dict devolve a linha no esquema FIELDNAMES.
    """
    if not os.path.isfile(src_csv):
        print(f"[Reaproveitamento] CSV de origem não encontrado: {src_csv}")
        return 0
    new_rows = []
    with open(src_csv, newline='') as f:
        for row in csv.DictReader(f):
            if not selector(row):
                continue
            out = transform(row)
            key = make_key(out['image_name'], out['num_gaussians'], out['lambda_rdo'], out['seed'], out['config'])
            if key in done_keys:
                continue
            new_rows.append(out)
            done_keys.add(key)
    append_rows(dst_csv, new_rows)
    return len(new_rows)

