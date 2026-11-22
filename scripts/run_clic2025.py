"""Execução do benchmark CLIC 2025 (30 imagens de teste ~2K em media/clic2025_image_test) comparando:
  1. Baseline: Image-GS original (14 bits, sem ARM) em N in {10k, 30k, 50k, 100k}
  2. Nosso método vencedor: GaussianARM (Shallow, C=2, stabilize_grads=True, STE 8 bits)
     em N in {10k, 30k, 50k, 100k} para 3 valores de lambda:
       - 0.0001 (melhor resultado geral)
       - 0.0005 (imediatamente inferior)
       - 0.0075 (pior / mais agressivo)

Tolerância a falhas e quedas de energia:
  - Antes de iniciar, lê o CSV de saída (resultados_diego_csvs/results_clic2025.csv) e pula
    todos os cenários já concluídos.
  - Caso uma queda de luz tenha interrompido o script logo após o main.py salvar o checkpoint (.pt)
    em results/, o script detecta e recupera automaticamente as métricas do checkpoint existente
    sem precisar treinar novamente.
  - Cada resultado é gravado linha a linha com flush + os.fsync imediato no disco,
    e a ordem de execução intercala por imagem (completa os 4 Ns do baseline e os 4 Ns dos 3 lambdas
    do nosso método para cada imagem antes de ir para a próxima), permitindo rodar a análise com
    Bootstrap (IC 95%) a qualquer momento sobre as imagens já concluídas.

Uso (a partir da raiz do repositório):
    python scripts/run_clic2025.py --dry-run
    nohup python -u scripts/run_clic2025.py --workers 2 > nohup_clic2025.out 2>&1 &
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import glob
import os
import shutil
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_runs import (FIELDNAMES, load_done_keys, make_key, num_pixels, run_one)  # noqa: E402
import torch  # noqa: E402

CLIC_IMAGES_DIR = "media/clic2025_image_test"
RESULTS_CSV = "resultados_diego_csvs/results_clic2025.csv"

MAX_STEPS = 50000
SEED = 123
NUM_GAUSSIANS_LIST = [10000, 30000, 50000, 100000]
DEFAULT_LAMBDAS = [0.0001, 0.0005, 0.0075]

BASELINE_CONFIG = "image_gs_14bits"
WINNER_ARCH = "shallow"
WINNER_CONTEXT = 2
WINNER_CONFIG = "shallow_c2_stab"
WINNER_ARM = dict(arch=WINNER_ARCH, context=WINNER_CONTEXT, use_ste=True, bits=8, stabilize=True)

_csv_lock = threading.Lock()


def list_clic_images(images_dir, subset=None):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    paths.sort()
    if subset:
        wanted = set(subset)
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in wanted]
    return paths


def append_row_fsync(csv_path, row):
    """Salva uma linha imediatamente no CSV e força gravação física no disco (os.fsync)."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())


def try_recover_from_checkpoint(img_path, exp_name, num_g, max_steps):
    """Se já existir um checkpoint salvo de antes de uma queda de energia, recupera as métricas sem retreinar."""
    ckpts = glob.glob(f"results/{exp_name}/**/*.pt", recursive=True)
    if not ckpts:
        return None
    try:
        d = torch.load(max(ckpts, key=os.path.getmtime), map_location='cpu', weights_only=False)
        if 'psnr' not in d or 'bytes' not in d or 'lpips' not in d:
            return None
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
    except Exception:
        return None


def build_base_row(name, method, config, n, lam):
    if method == 'baseline':
        return {
            'image_name': name, 'method': 'baseline', 'config': BASELINE_CONFIG,
            'num_gaussians': n, 'lambda_rdo': 0.0, 'seed': SEED,
            'arm_arch': 'none', 'arm_context': 0, 'use_ste': 'False', 'arm_bits': 0,
            'stabilize_grads': 'False', 'log_scale_min': '', 'log_scale_max': '', 'stab_disable': '',
        }
    return {
        'image_name': name, 'method': 'arm', 'config': config,
        'num_gaussians': n, 'lambda_rdo': lam, 'seed': SEED,
        'arm_arch': WINNER_ARCH, 'arm_context': WINNER_CONTEXT, 'use_ste': 'True', 'arm_bits': 8,
        'stabilize_grads': 'True', 'log_scale_min': -7.0, 'log_scale_max': 5.0, 'stab_disable': '',
    }


def execute_job(idx, total, job, csv_path, done_set, retries, cleanup_results):
    img_path, name, method, config, n, lam, exp_name, arm_cfg = job
    key = make_key(name, n, lam, SEED, config)

    with _csv_lock:
        if key in done_set:
            print(f"[{idx}/{total}] [PULADO - JÁ NO CSV] {name[:16]}... | {config} | N={n} | lambda={lam:g}")
            return True

    res = try_recover_from_checkpoint(img_path, exp_name, n, MAX_STEPS)
    recovered = res is not None
    if not recovered:
        print(f"\n[{idx}/{total}] Rodando: {name[:16]}... | {config} | N={n} | lambda={lam:g}")
        res = run_one(img_path, exp_name, n, lam, MAX_STEPS, seed=SEED, arm=arm_cfg, retries=retries)
        if res is None:
            print(f"[ERRO] Falha definitiva em {exp_name}")
            return False

    row = build_base_row(name, method, config, n, lam)
    row.update(res)

    with _csv_lock:
        if key not in done_set:
            append_row_fsync(csv_path, row)
            done_set.add(key)

    if cleanup_results and os.path.isdir(f"results/{exp_name}"):
        shutil.rmtree(f"results/{exp_name}", ignore_errors=True)

    tag = "[RECUPERADO DO CHECKPOINT]" if recovered else "-> Concluído"
    print(f"[{idx}/{total}] {tag}: {name[:16]}... | {config} | N={n} | L={lam:g} | PSNR {float(res['psnr']):.2f} dB | "
          f"bpp {float(res['bpp']):.4f} (img {res['bpp_img']} + arm {res['bpp_arm']}) | {res['time_s']}s")
    return True


def main():
    ap = argparse.ArgumentParser(description="Benchmark CLIC 2025: Image-GS Baseline vs. GaussianARM (Shallow, C=2)")
    ap.add_argument('--images-dir', default=CLIC_IMAGES_DIR, help="Diretório das imagens do CLIC 2025")
    ap.add_argument('--csv', default=RESULTS_CSV, help="Caminho do CSV de saída")
    ap.add_argument('--images', nargs='*', default=None, help="Subconjunto opcional de nomes de imagens")
    ap.add_argument('--n', nargs='*', type=int, default=None, help="Subconjunto opcional de N (default: 10k 30k 50k 100k)")
    ap.add_argument('--lambdas', nargs='*', type=float, default=None,
                    help="Lista de lambdas para shallow_c2_stab (default: 0.0001 0.0005 0.0075)")
    ap.add_argument('--only', choices=['baseline', 'ours'], default=None, help="Roda apenas 'baseline' ou apenas 'ours'")
    ap.add_argument('--workers', type=int, default=1, help="Número de processos simultâneos na GPU (ex: 1 ou 2)")
    ap.add_argument('--retries', type=int, default=1, help="Tentativas extras em caso de erro em uma execução")
    ap.add_argument('--cleanup-results', action='store_true', help="Remove a pasta results/<exp_name> após salvar no CSV para economizar disco")
    ap.add_argument('--dry-run', action='store_true', help="Apenas mostra o resumo de execuções feitas e pendentes")
    args = ap.parse_args()

    n_list = args.n or NUM_GAUSSIANS_LIST
    lambdas = args.lambdas or DEFAULT_LAMBDAS
    images = list_clic_images(args.images_dir, args.images)
    if not images:
        print(f"Nenhuma imagem encontrada em '{args.images_dir}'.")
        return

    done = load_done_keys(args.csv)

    # Intercala por imagem: para cada imagem, roda os 4 Ns do baseline e os 4 Ns de cada lambda do nosso método.
    pending = []
    total_grid = 0
    for p in images:
        name = os.path.splitext(os.path.basename(p))[0]
        if args.only in (None, 'baseline'):
            for n in n_list:
                total_grid += 1
                if make_key(name, n, 0.0, SEED, BASELINE_CONFIG) not in done:
                    exp = f"clic2025_{name}_base_g{n}_L0"
                    pending.append((p, name, 'baseline', BASELINE_CONFIG, n, 0.0, exp, None))
        if args.only in (None, 'ours'):
            for lam in lambdas:
                for n in n_list:
                    total_grid += 1
                    if make_key(name, n, lam, SEED, WINNER_CONFIG) not in done:
                        exp = f"clic2025_{name}_shc2stab_g{n}_L{lam:g}"
                        pending.append((p, name, 'arm', WINNER_CONFIG, n, lam, exp, WINNER_ARM))

    n_base_pend = sum(1 for j in pending if j[2] == 'baseline')
    n_ours_pend = sum(1 for j in pending if j[2] == 'arm')

    print("=" * 84)
    print(f"Benchmark CLIC 2025 — Diretório: {args.images_dir} ({len(images)} imagens)")
    print(f"CSV de saída (retomável): {args.csv}")
    print(f"Grade N={n_list} | Nosso método: {WINNER_CONFIG} | Lambdas={lambdas} | Workers={args.workers}")
    print(f"Total planejado: {total_grid} | Já salvos no CSV: {total_grid - len(pending)} | Pendentes: {len(pending)}")
    print(f"  - Baseline ({BASELINE_CONFIG}) pendentes : {n_base_pend}")
    for lam in lambdas:
        n_l = sum(1 for j in pending if j[2] == 'arm' and abs(j[5] - lam) < 1e-9)
        print(f"  - Nosso ({WINNER_CONFIG}, L={lam:<7g}) pendentes: {n_l}")
    print("=" * 84)

    if args.dry_run or not pending:
        return

    total_pend = len(pending)
    if args.workers <= 1:
        for i, job in enumerate(pending, 1):
            execute_job(i, total_pend, job, args.csv, done, args.retries, args.cleanup_results)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(execute_job, i, total_pend, job, args.csv, done, args.retries, args.cleanup_results)
                for i, job in enumerate(pending, 1)
            ]
            for fut in as_completed(futures):
                fut.result()

    print("\nExecução do CLIC 2025 finalizada! Rode `python scripts/calc_clic2025.py` para gerar gráficos e IC 95% (Bootstrap).")


if __name__ == "__main__":
    main()
