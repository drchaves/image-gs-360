"""Pipeline completo de experimentos finais em 3 etapas encadeadas + Baseline:

Etapa 1 (context)  : Testa contextos C in {1, 2, 3} (e completa eventuais faltantes em {4, 8, 16, 32})
                     fixando arch='shallow', stabilize_grads=True, STE 8 bits, N=100k e 3 lambdas.
                     Ao final, avalia todos os contextos C in {1, 2, 3, 4, 8, 16, 32} via BD-Rate
                     (curva média global, PSNR vs. BPP) e seleciona automaticamente o melhor contexto C*.
Etapa 2 (arch)     : Fixa o melhor contexto C* da Etapa 1 e avalia as 3 arquiteturas
                     ('original', 'shallow', 'deep') em N=100k e 3 lambdas (reaproveitando 'shallow' C*
                     já rodada na Etapa 1). Seleciona automaticamente o vencedor geral (arch*, C*).
Etapa 3 (final)    : Com o vencedor geral (arch*, C*), roda a grade completa N in {10k, 30k, 50k, 100k}
                     x 3 lambdas (reaproveitando N=100k das Etapas 1/2) + baseline Image-GS (14 bits).

Reaproveitamento automático:
  - results_shallow_stab_context_4_8_16_32.csv -> importa C in {4, 8, 16, 32} (shallow, stab, N=100k)
  - results_archs_stabilize_3_lambdas_45_images.csv -> importa original/shallow/deep em C=16 (stab, N=100k)

O script é 100% retomável: execuções já presentes no CSV de saída são puladas.

Uso (na raiz do repositório):
    python scripts/run_final_comparison.py --dry-run                 # mostra resumo e pendências por etapa
    python scripts/run_final_comparison.py --only context            # roda apenas Etapa 1 (C=1, 2, 3)
    python scripts/run_final_comparison.py --only arch               # roda apenas Etapa 2 (3 arquiteturas em C*)
    python scripts/run_final_comparison.py --only final              # roda apenas Etapa 3 (vencedor em 4 Ns x 3 lambdas)
    python scripts/run_final_comparison.py --only baseline           # roda apenas o baseline (Image-GS sem ARM)
    python scripts/run_final_comparison.py                           # roda tudo em sequência (Etapa 1 -> 2 -> 3)
Opções extras:
    --best-context 2   (força manualmente C* sem depender da eleição automática)
    --best-arch shallow (força manualmente arch* na Etapa 3)
    --images anime-1_2k photo-3_2k   --n 10000 30000   --retries 2
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_runs import (list_images, load_done_keys, make_key, append_rows, run_one,  # noqa: E402
                          import_from_csv)
import numpy as np  # noqa: E402


def bjontegaard_rate(rate1, psnr1, rate2, psnr2):
    log_rate1 = np.log10(rate1)
    log_rate2 = np.log10(rate2)
    try:
        deg1 = 2 if (len(psnr1) >= 3 and np.min(np.abs(np.diff(psnr1))) >= 0.02) else 1
        deg2 = 2 if (len(psnr2) >= 3 and np.min(np.abs(np.diff(psnr2))) >= 0.02) else 1
        p1 = np.polyfit(psnr1, log_rate1, deg1)
        p2 = np.polyfit(psnr2, log_rate2, deg2)
    except Exception:
        return float('nan')

    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    if max_psnr <= min_psnr:
        return float('nan')

    int_p1 = np.polyint(p1)
    int_p2 = np.polyint(p2)
    int1 = np.polyval(int_p1, max_psnr) - np.polyval(int_p1, min_psnr)
    int2 = np.polyval(int_p2, max_psnr) - np.polyval(int_p2, min_psnr)
    avg_diff = (int2 - int1) / (max_psnr - min_psnr)
    return (10**avg_diff - 1) * 100

RESULTS_CSV = "resultados_diego_csvs/results_final_comparison.csv"
CONTEXT_CSV = "resultados_diego_csvs/results_shallow_stab_context_4_8_16_32.csv"
ARCHS_CSV = "resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv"

MAX_STEPS = 50000          # mesmo valor dos experimentos de arquitetura/contexto (o treino termina antes pelo schedule de LR)
SEED = 123
N_ABLATION = 100000
NUM_GAUSSIANS_LIST = [10000, 30000, 50000, 100000]
RATE_LOSS_RATIOS = [0.0005, 0.0025, 0.0075]

INITIAL_ARCH = "shallow"
NEW_CONTEXTS = [1, 2, 3]
ALL_CONTEXTS = [1, 2, 3, 4, 8, 16, 32]
ARCHITECTURES = ["original", "shallow", "deep"]

BASELINE_CONFIG = "image_gs_14bits"
ARCH_SHORT = {"shallow": "sh", "original": "or", "deep": "dp"}


def config_name(arch, context):
    return f"{arch}_c{int(context)}_stab"


def exp_tag(arch, context):
    return f"{ARCH_SHORT.get(arch, arch[:2])}c{int(context)}stab"


def make_arm_dict(arch, context):
    return dict(arch=arch, context=int(context), use_ste=True, bits=8, stabilize=True)


def import_existing(done):
    total_imp = 0

    # 1) Importa C in {4, 8, 16, 32} (shallow, stabilize_grads=True, N=100k) do CSV de contexto
    def sel_ctx(r):
        return (r.get('arm_arch', '').strip() == 'shallow'
                and str(r.get('stabilize_grads', '')).strip().lower() == 'true'
                and str(r.get('num_gaussians', '')).strip() == str(N_ABLATION)
                and r.get('arm_context', '').strip() in {'4', '8', '16', '32'})

    def tr_ctx(r):
        ctx = int(r['arm_context'])
        arch = r['arm_arch'].strip()
        return {
            'image_name': r['image_name'], 'method': 'arm', 'config': config_name(arch, ctx),
            'num_gaussians': r['num_gaussians'], 'lambda_rdo': r['lambda_rdo'], 'seed': SEED,
            'arm_arch': arch, 'arm_context': ctx, 'use_ste': 'True', 'arm_bits': 8,
            'stabilize_grads': 'True', 'log_scale_min': -7.0, 'log_scale_max': 5.0, 'stab_disable': '',
            'max_steps': r.get('max_steps', ''), 'final_step': r.get('final_step', ''),
            'final_num_gaussians': r.get('final_num_gaussians', ''),
            'psnr': r['psnr'], 'ssim': r.get('ssim', ''), 'msssim': r.get('msssim', ''), 'lpips': r.get('lpips', ''),
            'bytes': r.get('bytes', ''), 'bpp_img': r['bpp_img'], 'bpp_arm': r['bpp_arm'], 'bpp': r['bpp'],
            'time_s': '',
        }

    total_imp += import_from_csv(CONTEXT_CSV, RESULTS_CSV, done, sel_ctx, tr_ctx)

    # 2) Importa original/shallow/deep em C=16 (stabilize_grads=True, N=100k) do CSV de arquiteturas
    def sel_arch(r):
        return (r.get('arm_arch', '').strip() in ARCHITECTURES
                and str(r.get('stabilize_grads', '')).strip().lower() == 'true'
                and str(r.get('num_gaussians', '')).strip() == str(N_ABLATION))

    def tr_arch(r):
        arch = r['arm_arch'].strip()
        ctx = int(r.get('arm_context', 16) or 16)
        return {
            'image_name': r['image_name'], 'method': 'arm', 'config': config_name(arch, ctx),
            'num_gaussians': r['num_gaussians'], 'lambda_rdo': r['lambda_rdo'], 'seed': SEED,
            'arm_arch': arch, 'arm_context': ctx, 'use_ste': 'True', 'arm_bits': 8,
            'stabilize_grads': 'True', 'log_scale_min': -7.0, 'log_scale_max': 5.0, 'stab_disable': '',
            'max_steps': r.get('max_steps', ''), 'final_step': r.get('final_step', ''),
            'final_num_gaussians': r.get('final_num_gaussians', ''),
            'psnr': r['psnr'], 'ssim': r.get('ssim', ''), 'msssim': r.get('msssim', ''), 'lpips': r.get('lpips', ''),
            'bytes': r.get('bytes', ''), 'bpp_img': r['bpp_img'], 'bpp_arm': r['bpp_arm'], 'bpp': r['bpp'],
            'time_s': '',
        }

    total_imp += import_from_csv(ARCHS_CSV, RESULTS_CSV, done, sel_arch, tr_arch)
    return total_imp


def load_arm_rows_n100k():
    """Lê todas as linhas do método ARM em N=100k já salvas em RESULTS_CSV."""
    rows = []
    if not os.path.isfile(RESULTS_CSV):
        return rows
    with open(RESULTS_CSV, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('method') != 'arm':
                continue
            if int(float(r.get('num_gaussians', 0) or 0)) != N_ABLATION:
                continue
            if str(r.get('stabilize_grads', '')).strip().lower() != 'true':
                continue
            rows.append({
                'image_name': r['image_name'].strip(),
                'config': r['config'].strip(),
                'arm_arch': r['arm_arch'].strip(),
                'arm_context': int(float(r['arm_context'])),
                'lambda_rdo': float(r['lambda_rdo']),
                'psnr': float(r['psnr']),
                'bpp': float(r['bpp']),
                'bpp_img': float(r['bpp_img']),
                'bpp_arm': float(r['bpp_arm']),
            })
    return rows


def evaluate_best_config(candidate_configs, anchor_config, label_title, min_images=36):
    """Avalia uma lista de configs em N=100k por BD-Rate sobre a curva média global (imagens completas em todos os candidatos).

    Retorna (winner_config, stats_dict) ou (None, complete_per_cfg) se ainda faltarem candidatos/imagens.
    """
    rows = load_arm_rows_n100k()
    by_cfg_img = {}
    for r in rows:
        cfg = r['config']
        if cfg not in candidate_configs:
            continue
        by_cfg_img.setdefault(cfg, {}).setdefault(r['image_name'], {})[round(r['lambda_rdo'], 6)] = r

    expected_lams = {round(l, 6) for l in RATE_LOSS_RATIOS}
    complete_per_cfg = {
        cfg: {img for img, lmap in by_cfg_img.get(cfg, {}).items() if expected_lams.issubset(lmap.keys())}
        for cfg in candidate_configs
    }

    # Imagens que já estão completas em TODOS os candidatos avaliados
    common_imgs = set.intersection(*(complete_per_cfg[c] for c in candidate_configs)) if candidate_configs else set()
    if len(common_imgs) < min_images or anchor_config not in complete_per_cfg:
        return None, complete_per_cfg

    def mean_curve(cfg, imgs):
        rates, psnrs = [], []
        for lam in sorted(RATE_LOSS_RATIOS, reverse=True):
            lkey = round(lam, 6)
            pts = [by_cfg_img[cfg][img][lkey] for img in imgs]
            rates.append(np.mean([p['bpp'] for p in pts]))
            psnrs.append(np.mean([p['psnr'] for p in pts]))
        order = np.argsort(psnrs)
        return np.array(rates)[order], np.array(psnrs)[order]

    r_anc, p_anc = mean_curve(anchor_config, common_imgs)
    print(f"\n==========================================================================")
    print(f" {label_title} (N={N_ABLATION}, {len(common_imgs)} imagens completas em todos)")
    print(f" Âncora de referência: {anchor_config}")
    print(f"==========================================================================")

    bd_results = {}
    for cfg in candidate_configs:
        r_c, p_c = mean_curve(cfg, common_imgs)
        bd = 0.0 if cfg == anchor_config else bjontegaard_rate(r_anc, p_anc, r_c, p_c)
        bd_results[cfg] = bd
        print(f"  * {cfg:18s} | BD-Rate: {bd:+6.2f}% | BPP médio: {r_c.mean():.4f} | PSNR médio: {p_c.mean():.2f} dB")

    winner = min(bd_results, key=lambda k: bd_results[k])
    print(f"--------------------------------------------------------------------------")
    print(f" -> VENCEDOR SELECIONADO: {winner} (BD-Rate: {bd_results[winner]:+.2f}% vs. {anchor_config})")
    print(f"==========================================================================\n")
    return winner, complete_per_cfg


def build_stage_jobs(kind, images, done, arch, ctx, n_list, lambdas):
    """Gera a lista de jobs pendentes para uma configuração específica."""
    pending = []
    for p in images:
        name = os.path.splitext(os.path.basename(p))[0]
        if kind == 'baseline':
            cfg = BASELINE_CONFIG
            for n in n_list:
                if make_key(name, n, 0.0, SEED, cfg) not in done:
                    pending.append(('baseline', p, name, n, 0.0, cfg, None))
        else:
            cfg = config_name(arch, ctx)
            arm = make_arm_dict(arch, ctx)
            for n in n_list:
                for lam in lambdas:
                    if make_key(name, n, lam, SEED, cfg) not in done:
                        pending.append((kind, p, name, n, lam, cfg, arm))
    return pending


def execute_pending(pending, done, stage_label, retries=1):
    if not pending:
        print(f"[{stage_label}] Nenhum job pendente (tudo já concluído no CSV).")
        return
    print(f"\n[{stage_label}] Iniciando execução de {len(pending)} jobs pendentes...")
    for i, (kind, p, name, n, lam, cfg, arm) in enumerate(pending, 1):
        tag = 'base' if kind == 'baseline' else exp_tag(arm['arch'], arm['context'])
        exp = f"final_{name}_{tag}_g{n}_L{lam:g}"
        print(f"\n[{stage_label} {i}/{len(pending)}] {name} | {cfg} | N={n} | lambda={lam:g}")
        res = run_one(p, exp, n, lam, MAX_STEPS, seed=SEED, arm=arm, retries=retries)
        if res is None:
            continue
        row = {
            'image_name': name,
            'method': 'baseline' if kind == 'baseline' else 'arm',
            'config': cfg,
            'num_gaussians': n,
            'lambda_rdo': lam,
            'seed': SEED,
            'arm_arch': '-' if kind == 'baseline' else arm['arch'],
            'arm_context': '-' if kind == 'baseline' else arm['context'],
            'use_ste': '-' if kind == 'baseline' else 'True',
            'arm_bits': '-' if kind == 'baseline' else arm['bits'],
            'stabilize_grads': '-' if kind == 'baseline' else 'True',
            'log_scale_min': '' if kind == 'baseline' else -7.0,
            'log_scale_max': '' if kind == 'baseline' else 5.0,
            'stab_disable': '',
        }
        row.update(res)
        append_rows(RESULTS_CSV, [row])
        done.add(make_key(name, n, lam, SEED, cfg))
        print(f"-> PSNR {float(res['psnr']):.2f} | bpp {float(res['bpp']):.4f} (img {res['bpp_img']} + arm {res['bpp_arm']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['context', 'arch', 'final', 'baseline', 'method'], default=None,
                    help="Executa apenas uma etapa específica ('context'=Etapa 1, 'arch'=Etapa 2, 'final'/'method'=Etapa 3, 'baseline'=Image-GS).")
    ap.add_argument('--best-context', type=int, default=None,
                    help="Fixa manualmente o melhor contexto C* para as Etapas 2 e 3 (se omitido, elege automaticamente via BD-Rate).")
    ap.add_argument('--best-arch', choices=ARCHITECTURES, default=None,
                    help="Fixa manualmente a melhor arquitetura arch* para a Etapa 3 (se omitido, elege automaticamente via BD-Rate).")
    ap.add_argument('--images', nargs='*', default=None)
    ap.add_argument('--n', nargs='*', type=int, default=None, help="subconjunto de num_gaussians para a Etapa 3 / baseline")
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    n_list = args.n or NUM_GAUSSIANS_LIST
    images = list_images(args.images)
    done = load_done_keys(RESULTS_CSV)
    imp = import_existing(done)
    if imp:
        print(f"[Reaproveitamento] {imp} execuções existentes (N=100k) importadas para {RESULTS_CSV}")

    # =========================================================================
    # ETAPA 1: Contextos C in {1, 2, 3} (e eventuais faltantes de {4, 8, 16, 32}) com Shallow em N=100k
    # =========================================================================
    stage1_pending = []
    for ctx in ALL_CONTEXTS:
        stage1_pending.extend(
            build_stage_jobs('context', images, done, INITIAL_ARCH, ctx, [N_ABLATION], RATE_LOSS_RATIOS)
        )

    if args.only in (None, 'context'):
        print(f"[Etapa 1 - Contextos {ALL_CONTEXTS} c/ {INITIAL_ARCH}, N={N_ABLATION}] Pendentes: {len(stage1_pending)}")
        if not args.dry_run:
            execute_pending(stage1_pending, done, "Etapa 1 (Contexto)", retries=args.retries)

    # Eleição automática (ou manual via --best-context) do melhor contexto C*
    ctx_candidates = [config_name(INITIAL_ARCH, c) for c in ALL_CONTEXTS]
    winner_ctx_cfg, complete_ctx_map = evaluate_best_config(
        ctx_candidates,
        anchor_config=config_name(INITIAL_ARCH, 16),
        label_title="ETAPA 1: ELEIÇÃO DO MELHOR CONTEXTO (Arquitetura Shallow)"
    )

    if args.best_context is not None:
        best_ctx = args.best_context
        print(f"[Configuração] Usando --best-context={best_ctx} definido manualmente.")
    elif winner_ctx_cfg is not None:
        best_ctx = int(winner_ctx_cfg.split('_c')[1].split('_')[0])
    else:
        best_ctx = None
        if args.only in (None, 'arch', 'final', 'method'):
            print("[Aviso Etapa 1 -> 2] Ainda existem contextos incompletos na Etapa 1:")
            for c in ALL_CONTEXTS:
                cfg = config_name(INITIAL_ARCH, c)
                print(f"  - C={c:<2d} ({cfg}): {len(complete_ctx_map.get(cfg, set()))}/45 imagens completas")
            if args.dry_run:
                print("  (Em --dry-run: assumindo provisoriamente C*=4 apenas para estimar os jobs das Etapas 2 e 3)")
                best_ctx = 4
            elif args.only == 'context':
                return
            else:
                print("  Conclua a Etapa 1 primeiro ou informe --best-context C manualmente.")
                return

    if args.only == 'context':
        return

    # =========================================================================
    # ETAPA 2: Fixa C = C* e varia as 3 arquiteturas (original, shallow, deep) em N=100k
    # =========================================================================
    stage2_pending = []
    for arch in ARCHITECTURES:
        stage2_pending.extend(
            build_stage_jobs('arch', images, done, arch, best_ctx, [N_ABLATION], RATE_LOSS_RATIOS)
        )

    if args.only in (None, 'arch'):
        print(f"[Etapa 2 - Arquiteturas {ARCHITECTURES} fixando C*={best_ctx}, N={N_ABLATION}] Pendentes: {len(stage2_pending)}")
        if not args.dry_run:
            execute_pending(stage2_pending, done, f"Etapa 2 (Arquiteturas @ C*={best_ctx})", retries=args.retries)

    arch_candidates = [config_name(a, best_ctx) for a in ARCHITECTURES]
    winner_arch_cfg, complete_arch_map = evaluate_best_config(
        arch_candidates,
        anchor_config=config_name("original", best_ctx),
        label_title=f"ETAPA 2: ELEIÇÃO DA MELHOR ARQUITETURA FIXANDO C*={best_ctx}"
    )

    if args.best_arch is not None:
        best_arch = args.best_arch
        print(f"[Configuração] Usando --best-arch={best_arch} definido manualmente.")
    elif winner_arch_cfg is not None:
        best_arch = winner_arch_cfg.split('_c')[0]
    else:
        best_arch = None
        if args.only in (None, 'final', 'method'):
            print(f"[Aviso Etapa 2 -> 3] Ainda existem arquiteturas incompletas em C*={best_ctx}:")
            for a in ARCHITECTURES:
                cfg = config_name(a, best_ctx)
                print(f"  - {a:8s} ({cfg}): {len(complete_arch_map.get(cfg, set()))}/45 imagens completas")
            if args.dry_run:
                print(f"  (Em --dry-run: assumindo provisoriamente arch*='shallow' e C*={best_ctx} para estimar a Etapa 3)")
                best_arch = "shallow"
            elif args.only == 'arch':
                return
            else:
                print("  Conclua a Etapa 2 primeiro ou informe --best-arch e --best-context manualmente.")
                return

    if args.only == 'arch':
        return

    # =========================================================================
    # ETAPA 3: Vencedor Geral (arch*, C*) variando N in {10k, 30k, 50k, 100k} x 3 lambdas + Baseline
    # =========================================================================
    stage3_baseline_pending = []
    stage3_winner_pending = []

    if args.only in (None, 'baseline'):
        stage3_baseline_pending = build_stage_jobs('baseline', images, done, None, None, n_list, [0.0])

    if args.only in (None, 'final', 'method'):
        stage3_winner_pending = build_stage_jobs('final', images, done, best_arch, best_ctx, n_list, RATE_LOSS_RATIOS)

    winner_cfg_str = config_name(best_arch, best_ctx) if best_arch and best_ctx else "N/A"
    print(f"[Etapa 3 - Comparação Final | Vencedor: {winner_cfg_str} em N={n_list}] "
          f"Pendentes -> baseline: {len(stage3_baseline_pending)} | vencedor ({winner_cfg_str}): {len(stage3_winner_pending)}")

    if args.dry_run:
        return

    if stage3_baseline_pending:
        execute_pending(stage3_baseline_pending, done, "Etapa 3 (Baseline)", retries=args.retries)
    if stage3_winner_pending:
        execute_pending(stage3_winner_pending, done, f"Etapa 3 (Vencedor {winner_cfg_str})", retries=args.retries)


if __name__ == "__main__":
    main()


