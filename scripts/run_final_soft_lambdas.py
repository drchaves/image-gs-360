"""Completa os lambdas suaves (0.0001, 0.0005, 0.0010) para o vencedor geral (Shallow, C=2, estabilizado)
variando N in {10k, 30k, 50k, 100k} nas 45 imagens.

Como lambda=0.0005 e o baseline (Image-GS original) já estão 100% prontos em
resultados_diego_csvs/results_final_comparison.csv, este script pula automaticamente
tudo que já existe e roda apenas os 2 novos valores suaves de lambda (0.0001 e 0.0010),
totalizando 360 execuções (45 imagens x 4 valores de N x 2 novos lambdas).

Uso (na raiz do repositório):
    python scripts/run_final_soft_lambdas.py --dry-run
    nohup python -u scripts/run_final_soft_lambdas.py > nohup_soft_lambdas.out 2>&1 &
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_runs import (list_images, load_done_keys, make_key, append_rows, run_one)  # noqa: E402

RESULTS_CSV = "resultados_diego_csvs/results_final_comparison.csv"

MAX_STEPS = 50000
SEED = 123
NUM_GAUSSIANS_LIST = [10000, 30000, 50000, 100000]
SOFT_LAMBDAS = [0.0001, 0.0005, 0.0010]

WINNER_ARCH = "shallow"
WINNER_CONTEXT = 2
WINNER_CONFIG = "shallow_c2_stab"
WINNER_ARM = dict(arch=WINNER_ARCH, context=WINNER_CONTEXT, use_ste=True, bits=8, stabilize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='*', default=None, help="subconjunto opcional de imagens")
    ap.add_argument('--n', nargs='*', type=int, default=None, help="subconjunto opcional de num_gaussians")
    ap.add_argument('--lambdas', nargs='*', type=float, default=None, help="subconjunto opcional de lambdas")
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    n_list = args.n or NUM_GAUSSIANS_LIST
    lam_list = args.lambdas or SOFT_LAMBDAS
    images = list_images(args.images)
    done = load_done_keys(RESULTS_CSV)

    pending = []
    for p in images:
        name = os.path.splitext(os.path.basename(p))[0]
        for n in n_list:
            for lam in lam_list:
                if make_key(name, n, lam, SEED, WINNER_CONFIG) not in done:
                    pending.append((p, name, n, lam))

    total_grid = len(images) * len(n_list) * len(lam_list)
    print(f"Configuração vencedora: {WINNER_CONFIG} | N={n_list} | lambdas={lam_list}")
    print(f"Total na grade: {total_grid} | Já concluídos no CSV: {total_grid - len(pending)} | Pendentes: {len(pending)}")
    for lam in lam_list:
        c_lam = sum(1 for j in pending if abs(j[3] - lam) < 1e-9)
        print(f"  - lambda={lam:<7g}: {c_lam} pendentes")

    if args.dry_run or not pending:
        return

    for i, (p, name, n, lam) in enumerate(pending, 1):
        exp = f"final_{name}_shc2stab_g{n}_L{lam:g}"
        print(f"\n[{i}/{len(pending)}] {name} | {WINNER_CONFIG} | N={n} | lambda={lam:g}")
        res = run_one(p, exp, n, lam, MAX_STEPS, seed=SEED, arm=WINNER_ARM, retries=args.retries)
        if res is None:
            continue
        row = {
            'image_name': name,
            'method': 'arm',
            'config': WINNER_CONFIG,
            'num_gaussians': n,
            'lambda_rdo': lam,
            'seed': SEED,
            'arm_arch': WINNER_ARCH,
            'arm_context': WINNER_CONTEXT,
            'use_ste': 'True',
            'arm_bits': 8,
            'stabilize_grads': 'True',
            'log_scale_min': -7.0,
            'log_scale_max': 5.0,
            'stab_disable': '',
        }
        row.update(res)
        append_rows(RESULTS_CSV, [row])
        done.add(make_key(name, n, lam, SEED, WINNER_CONFIG))
        print(f"-> PSNR {float(res['psnr']):.2f} | bpp {float(res['bpp']):.4f} (img {res['bpp_img']} + arm {res['bpp_arm']})")


if __name__ == "__main__":
    main()
