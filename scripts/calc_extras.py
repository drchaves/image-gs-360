"""Análise dos experimentos complementares (run_extras.py): sensibilidade do limite inferior da escala,
ablação da estabilização e variabilidade entre sementes.

Entrada : resultados_diego_csvs/results_extras.csv
Saída   : plots_extras/*.pdf|png  e  plots_extras/tabelas_extras.tex

BD-Rate sobre o PSNR, calculado nas curvas médias (3 pontos, um por lambda) sobre o subconjunto de imagens;
IC95% por bootstrap sobre imagens. Convenção: BD-Rate de cada variante *em relação à configuração completa*
('full'): valor POSITIVO = a variante é pior (a medida removida/alterada ajudava).

Uso: python scripts/calc_extras.py [--csv CAMINHO] [--out DIR] [--boot 1000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _analysis_utils import fnum, pm, complete_images, to_matrix, bootstrap_bd  # noqa: E402

LAMBDAS = [0.0005, 0.0025, 0.0075]
GROUPS = {
    'sensibilidade': (["full", "ls_min_-9", "ls_min_-11"],
                      {"full": "$-7$ (padrão)", "ls_min_-9": "$-9$", "ls_min_-11": "$-11$"}),
    'ablacao': (["full", "none", "no_surrogate", "no_clamp", "no_clip", "no_ste", "no_adam"],
                {"full": "Completa", "none": "Nenhuma (sem estabilização)", "no_surrogate": "Sem gradiente substituto",
                 "no_clamp": "Sem limite de escala", "no_clip": "Sem recorte de norma",
                 "no_ste": "Sem STE com retorno à faixa", "no_adam": "Sem preservação dos momentos do Adam"}),
    'sementes': (["full", "full_seed7", "full_seed42"],
                 {"full": "123", "full_seed7": "7", "full_seed42": "42"}),
}


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze_group(df, configs, boot):
    """Devolve (imagens, {cfg: (R,P)}, {cfg: (bd, lo, hi)}) restrito às imagens completas em todas as configs do grupo."""
    present = [c for c in configs if (df.config == c).any()]
    if 'full' not in present or len(present) < 2:
        return None
    imgs = None
    for c in present:
        ok = set(complete_images(df[df.config == c], 'lambda_rdo', LAMBDAS))
        imgs = ok if imgs is None else imgs & ok
    imgs = sorted(imgs)
    if not imgs:
        return None
    mats = {c: to_matrix(df[df.config == c], imgs, 'lambda_rdo', LAMBDAS) for c in present}
    Rf, Pf = mats['full']
    bds = {}
    for c in present:
        if c == 'full':
            bds[c] = (0.0, 0.0, 0.0)
        else:
            bds[c] = bootstrap_bd(Rf, Pf, *mats[c], n_boot=boot)
    return imgs, mats, bds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default="resultados_diego_csvs/results_extras.csv")
    ap.add_argument('--out', default="plots_extras")
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})

    df = pd.read_csv(args.csv)
    for c in ['num_gaussians', 'lambda_rdo', 'psnr', 'bpp', 'bpp_img', 'bpp_arm', 'seed']:
        df[c] = df[c].astype(float)
    tex = []

    # ---------------------------------------------------------------- sensibilidade e ablação
    for gname, caption_tag in [('sensibilidade', 'sens'), ('ablacao', 'abl')]:
        configs, names = GROUPS[gname]
        res = analyze_group(df, configs, args.boot)
        if res is None:
            print(f"[{gname}] dados insuficientes ainda.")
            continue
        imgs, mats, bds = res
        print(f"\n=== {gname.upper()} ({len(imgs)} imagens) — BD-Rate (PSNR) vs. configuração completa; + = pior ===")
        rows = []
        for c in configs:
            if c not in mats:
                continue
            R, P = mats[c]
            pt, lo, hi = bds[c]
            rows.append((c, R.mean(), P.mean(), pt, lo, hi))
            print(f"  {c:14s}: bpp {R.mean():.4f} | PSNR {P.mean():.2f} dB | BD {pt:+6.2f}%  IC95% [{lo:+.2f}; {hi:+.2f}]"
                  + ("" if c == 'full' or lo > 0 or hi < 0 else "   (IC inclui 0)"))
        # gráfico
        if gname == 'sensibilidade':
            fig, ax = plt.subplots(figsize=(8, 5.8))
            for c, col in zip(configs, ["#d62728", "#1f77b4", "#2ca02c"]):
                if c in mats:
                    R, P = mats[c]
                    ax.plot(R.mean(0), P.mean(0), '-o', color=col, lw=2, ms=7, label=f"$s_{{\\min}}$ = {names[c].replace('$', '')}  [BD: {bds[c][0]:+.2f}%]")
            ax.set_xlabel("Taxa total — bpp"); ax.set_ylabel("PSNR médio (dB)"); ax.grid(True, ls="--", alpha=0.6)
            ax.set_title(f"Sensibilidade do limite inferior da escala — {len(imgs)} imagens", fontweight="bold"); ax.legend(fontsize=9)
            save(fig, args.out, "sensibilidade_limite_escala")
        else:
            labs = [c for c in configs if c in mats and c != 'full']
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            y = np.arange(len(labs))
            pts = np.array([bds[c][0] for c in labs]); lo = np.array([bds[c][1] for c in labs]); hi = np.array([bds[c][2] for c in labs])
            ax.barh(y, pts, xerr=[np.maximum(pts - lo, 0), np.maximum(hi - pts, 0)], color="#1f77b4", alpha=0.85, capsize=4)
            ax.set_yticks(y); ax.set_yticklabels([names[c].replace('$', '') for c in labs]); ax.invert_yaxis()
            ax.axvline(0, color='k', lw=1); ax.set_xlabel("BD-Rate (%) em relação à configuração completa (positivo = pior)")
            ax.set_title(f"Ablação da estabilização — {len(imgs)} imagens", fontweight="bold"); ax.grid(True, axis='x', ls="--", alpha=0.6)
            save(fig, args.out, "ablacao_estabilizacao")
        # tabela
        cap = ("Sensibilidade ao limite inferior $s_{\\min}$ da ativação de escala (limite superior fixo em $5$). "
               if gname == 'sensibilidade' else
               "Ablação da estabilização: cada linha desliga um componente (\\textit{leave-one-out}). ")
        cap += (f"BD-Rate (PSNR) em relação à configuração completa (valores positivos indicam piora); IC de 95\\% por \\textit{{bootstrap}} sobre "
                f"{len(imgs)} imagens. \\textit{{Shallow}}, $C=4$, $N=100.000$, três valores de $\\lambda$.")
        t = ["\\begin{table}[htbp]\n    \\centering", f"    \\caption{{{cap}}}", f"    \\label{{tab:{'sensibilidade_escala' if gname=='sensibilidade' else 'ablacao_estab'}}}",
             "    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{|l|c|c|c|}\n        \\hline",
             "        \\textbf{" + ("$s_{\\min}$" if gname == 'sensibilidade' else "Configuração") + "} & \\textbf{bpp ($\\downarrow$)} & \\textbf{PSNR (dB) ($\\uparrow$)} & \\textbf{BD-Rate [IC 95\\%]} \\\\ \\hline\\hline"]
        for c, b_, p_, pt, lo, hi in rows:
            bd_txt = "---" if c == 'full' else f"{pm(pt)} [{fnum(lo, 2, True)}; {fnum(hi, 2, True)}]"
            t.append(f"        {names[c]} & {fnum(b_, 4)} & {fnum(p_)} & {bd_txt} \\\\ \\hline")
        t.append("    \\end{tabular}\n    }\n    \\fonte O autor (2026).\n\\end{table}\n")
        tex.append("\n".join(t))

    # ---------------------------------------------------------------- sementes
    configs, names = GROUPS['sementes']
    present = [c for c in configs if (df.config == c).any()]
    if len(present) >= 2:
        imgs = None
        for c in present:
            ok = set(complete_images(df[df.config == c], 'lambda_rdo', LAMBDAS))
            imgs = ok if imgs is None else imgs & ok
        imgs = sorted(imgs)
        mats = {c: to_matrix(df[df.config == c], imgs, 'lambda_rdo', LAMBDAS) for c in present}
        print(f"\n=== SEMENTES ({len(imgs)} imagens, {len(present)} sementes) ===")
        t = ["\\begin{table}[htbp]\n    \\centering",
             f"    \\caption{{Variabilidade entre sementes ($\\textit{{Shallow}}$, $C=4$, $N=100.000$; {len(imgs)} imagens). Médias sobre as imagens por $\\lambda$; "
             "BD-Rate (PSNR) de cada semente em relação à semente 123.}}",
             "    \\label{tab:sementes}\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{|c|c|c|c|c|}\n        \\hline",
             "        \\textbf{Semente} & $\\boldsymbol{\\lambda}$ & \\textbf{bpp} & \\textbf{PSNR (dB)} & \\textbf{BD-Rate vs. semente 123} \\\\ \\hline\\hline"]
        Rf, Pf = mats['full']
        bd_seed = {}
        for c in present:
            bd_seed[c] = 0.0 if c == 'full' else bootstrap_bd(Rf, Pf, *mats[c], n_boot=args.boot)
        for c in present:
            R, P = mats[c]
            for k, lam in enumerate(LAMBDAS):
                first = (f"\\multirow{{3}}{{*}}{{{names[c]}}}" if k == 0 else "")
                bdtxt = ""
                if k == 0:
                    bdtxt = "\\multirow{3}{*}{---}" if c == 'full' else f"\\multirow{{3}}{{*}}{{{pm(bd_seed[c][0])}}}"
                sep = "\\cline{2-4}" if k < 2 else "\\hline"
                t.append(f"        {first} & {fnum(lam, 4)} & {fnum(R[:, k].mean(), 4)} & {fnum(P[:, k].mean())} & {bdtxt} \\\\ {sep}")
        t.append("    \\end{tabular}\n    }\n    \\fonte O autor (2026).\n\\end{table}\n")
        tex.append("\n".join(t))
        for c in present:
            R, P = mats[c]
            v = bd_seed[c]
            print(f"  semente {names[c]:>3s}: bpp médio {R.mean():.4f} | PSNR médio {P.mean():.2f} dB"
                  + ("" if c == 'full' else f" | BD vs 123: {v[0]:+.2f}% [{v[1]:+.2f}; {v[2]:+.2f}]"))
        # desvio-padrão entre sementes por lambda
        for k, lam in enumerate(LAMBDAS):
            ps = [mats[c][1][:, k].mean() for c in present]; bs = [mats[c][0][:, k].mean() for c in present]
            print(f"  lambda={lam:<7g}: desvio entre sementes — PSNR {np.std(ps, ddof=1):.3f} dB | bpp {np.std(bs, ddof=1):.4f}")
        noise = max(abs(bd_seed[c][0]) for c in present if c != 'full')
        print(f"  => piso de ruído do BD-Rate (maior |BD| entre sementes): {noise:.2f}%. "
              "Diferenças menores que isso, nas demais comparações, não devem ser tratadas como efeito real.")
    else:
        print("\n[sementes] dados insuficientes ainda.")

    if tex:
        with open(os.path.join(args.out, "tabelas_extras.tex"), "w") as f:
            f.write("\n".join(tex))
        print(f"\nSalvo em {args.out}/ (tabelas_extras.tex e figuras)")


if __name__ == "__main__":
    main()

