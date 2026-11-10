"""Experimentos complementares (todos com a configuração final: Shallow, C=4, estabilizada, STE 8 bits, N=100k)
sobre um subconjunto de imagens (2 por categoria):

  --exp sensitivity : limite inferior do clamp da ativação de escala s_t em {-7 (já existe, importado), -9, -11}
  --exp ablation    : ablação "leave-one-out" da estabilização:
                      none (stabilize_grads desligado), no_surrogate, no_clamp, no_clip, no_ste, no_adam
  --exp seeds       : repetição da configuração completa com sementes 7 e 42 (123 já existe, importada)
  --exp all         : os três

A configuração de referência ('full': clamp [-7,5], todos os componentes, semente 123) é importada
de results_shallow_stab_context_4_8_16_32.csv (linhas C=4, N=100k) para as imagens do subconjunto.

Retomável (pula o que já está no CSV). Uso, na raiz do repositório:
    python scripts/run_extras.py --exp sensitivity --dry-run
    python scripts/run_extras.py --exp sensitivity       # ~60 execuções  (prioridade: rode primeiro)
    python scripts/run_extras.py --exp ablation          # ~180 execuções
    python scripts/run_extras.py --exp seeds             # ~60 execuções
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_runs import (list_images, load_done_keys, make_key, append_rows, run_one,  # noqa: E402
                          import_from_csv)

RESULTS_CSV = "resultados_diego_csvs/results_extras.csv"
CONTEXT_CSV = "resultados_diego_csvs/results_shallow_stab_context_4_8_16_32.csv"

# 2 imagens por categoria (escolha arbitrária e fixa; evita art-6_2k, que teve falha de execução)
SUBSET = ["anime-2_2k", "anime-7_2k", "art-2_2k", "art-7_2k", "painting-2_2k", "painting-7_2k",
          "photo-2_2k", "photo-7_2k", "vector-2_2k", "vector-7_2k"]
MAX_STEPS = 50000
N = 100000
LAMBDAS = [0.0005, 0.0025, 0.0075]
BASE_ARM = dict(arch="shallow", context=4, use_ste=True, bits=8, stabilize=True)
REF_SEED = 123


def experiments(which):
    """Devolve lista de (exp, config, seed, arm_dict)."""
    out = []
    if which in ('sensitivity', 'all'):
        for lo in (-9, -11):
            out.append(('sensitivity', f"ls_min_{lo}", REF_SEED, dict(BASE_ARM, log_scale_min=lo)))
    if which in ('ablation', 'all'):
        out.append(('ablation', "none", REF_SEED, dict(BASE_ARM, stabilize=False)))
        for comp in ('surrogate', 'clamp', 'clip', 'ste', 'adam'):
            out.append(('ablation', f"no_{comp}", REF_SEED, dict(BASE_ARM, disable=[comp])))
    if which in ('seeds', 'all'):
        for s in (7, 42):
            out.append(('seeds', f"full_seed{s}", s, dict(BASE_ARM)))
    return out


def import_reference(done):
    def sel(r):
        return (r.get('arm_context', '').strip() == '4' and r.get('arm_arch', '').strip() == 'shallow'
                and str(r.get('stabilize_grads', '')).strip().lower() == 'true'
                and str(r.get('num_gaussians', '')).strip() == str(N) and r['image_name'] in SUBSET)

    def tr(r):
        return {
            'image_name': r['image_name'], 'method': 'arm', 'config': 'full',
            'num_gaussians': r['num_gaussians'], 'lambda_rdo': r['lambda_rdo'], 'seed': REF_SEED,
            'arm_arch': 'shallow', 'arm_context': 4, 'use_ste': 'True', 'arm_bits': 8,
            'stabilize_grads': 'True', 'log_scale_min': -7.0, 'log_scale_max': 5.0, 'stab_disable': '',
            'max_steps': r.get('max_steps', ''), 'final_step': r.get('final_step', ''),
            'final_num_gaussians': r.get('final_num_gaussians', ''),
            'psnr': r['psnr'], 'ssim': r.get('ssim', ''), 'msssim': r.get('msssim', ''), 'lpips': r.get('lpips', ''),
            'bytes': r.get('bytes', ''), 'bpp_img': r['bpp_img'], 'bpp_arm': r['bpp_arm'], 'bpp': r['bpp'],
            'time_s': '',
        }
    return import_from_csv(CONTEXT_CSV, RESULTS_CSV, done, sel, tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=['sensitivity', 'ablation', 'seeds', 'all'], required=True)
    ap.add_argument('--images', nargs='*', default=None, help="substitui o subconjunto padrão")
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    images = list_images(args.images or SUBSET)
    done = load_done_keys(RESULTS_CSV)
    imp = import_reference(done)
    if imp:
        print(f"[Reaproveitamento] {imp} execuções de referência ('full') importadas de {CONTEXT_CSV}")

    pending = []
    for exp, cfg, seed, arm in experiments(args.exp):
        for p in images:
            name = os.path.splitext(os.path.basename(p))[0]
            for lam in LAMBDAS:
                if make_key(name, N, lam, seed, cfg) not in done:
                    pending.append((exp, cfg, seed, arm, p, name, lam))

    print(f"Imagens: {len(images)} | pendentes: {len(pending)}")
    if args.dry_run:
        for exp, cfg in sorted({(j[0], j[1]) for j in pending}):
            print(f"  {exp:12s} {cfg:16s}: {sum(1 for j in pending if j[1] == cfg)}")
        return

    for i, (exp, cfg, seed, arm, p, name, lam) in enumerate(pending, 1):
        exp_name = f"extras_{name}_{cfg}_g{N}_L{lam:g}"
        print(f"\n[{i}/{len(pending)}] {exp} | {cfg} | {name} | lambda={lam:g} | seed={seed}")
        res = run_one(p, exp_name, N, lam, MAX_STEPS, seed=seed, arm=arm, retries=args.retries)
        if res is None:
            continue
        row = {
            'image_name': name, 'method': 'arm', 'config': cfg, 'num_gaussians': N, 'lambda_rdo': lam, 'seed': seed,
            'arm_arch': arm['arch'], 'arm_context': arm['context'], 'use_ste': 'True', 'arm_bits': arm['bits'],
            'stabilize_grads': str(arm['stabilize']),
            'log_scale_min': arm.get('log_scale_min', -7.0), 'log_scale_max': arm.get('log_scale_max', 5.0),
            'stab_disable': ' '.join(arm.get('disable', [])),
        }
        row.update(res)
        append_rows(RESULTS_CSV, [row])
        done.add(make_key(name, N, lam, seed, cfg))
        print(f"-> PSNR {float(res['psnr']):.2f} | bpp {float(res['bpp']):.4f}")


if __name__ == "__main__":
    main()

