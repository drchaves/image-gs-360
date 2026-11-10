"""Diagramas 3D (estilo VGG) das arquiteturas da GaussianARM.

Estilo visual alinhado à figura "Pipeline de entropia autorregressivo":
fonte serifada (Times-like), caixas pastel com bordas coloridas e
contêiner tracejado.

Uso: conda activate image-gs && python draw_arm_architectures_3d.py
Gera arm_arch_3d_{original,shallow,deep}.pdf na raiz.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch

# Liberation Serif é métrica-compatível com Times New Roman
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

F = 8  # features por gaussiana
ARCHS = {
    "original": [(8, 64, 16), (64, 64, 1), (64, 16, 1)],
    "shallower": [(8, 93, 16), (93, 16, 1)],
    "deeper": [(8, 53, 16), (53, 53, 1), (53, 53, 1), (53, 16, 1)],
}

GREEN, GREEN_BG = "#1b7a5a", "#d9eee6"
PURPLE, PURPLE_BG = "#5a4bb5", "#e6e4f6"
ORANGE, ORANGE_BG = "#c0601a", "#f6d9c2"
GOLD, GOLD_BG = "#b8892b", "#f3e6c4"
TEXT, GRAY = "#333333", "#555555"

W, D = 1.0, 3.0           # espessura (eixo x) e profundidade (sequência N)
DX, DY = 0.45 * D, 0.30 * D


def height(c):
    return 0.12 * c + 0.4


def cuboid(ax, x, yc, c, face, edge, w=W):
    """Cubóide em projeção oblíqua; altura proporcional aos canais; centrado em yc."""
    h = height(c)
    y = yc - h / 2
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + w, y + h), (x + w + DX, y + h + DY), (x + DX, y + h + DY)]
    side = [(x + w, y), (x + w + DX, y + DY), (x + w + DX, y + h + DY), (x + w, y + h)]
    for poly, alpha in ((front, 1.0), (top, 0.65), (side, 0.85)):
        ax.add_patch(Polygon(poly, closed=True, facecolor=face, edgecolor=edge,
                             lw=1.4, alpha=alpha, joinstyle="round"))
    return x + w + DX


def caption(ax, x, y, title, lines):
    ax.text(x, y, title, ha="center", va="top", fontsize=12, color=TEXT)
    for i, ln in enumerate(lines):
        ax.text(x, y - 1.15 * (i + 1), ln, ha="center", va="top", fontsize=9.5, color=GRAY)


def arrow(ax, x1, x2, y=0.0, color=TEXT):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))


def fmt(n):
    return format(n, ",").replace(",", ".")


def draw(name, layers):
    fig, ax = plt.subplots(figsize=(3.9 * (len(layers) + 3), 6.4))
    total = sum(ci * co * k + co for ci, co, k in layers)
    gap, base = 3.6, 0.0
    ymin = -height(93) / 2 - 2.0
    cap_y = ymin + 0.2

    x = 0.0
    xe = cuboid(ax, x, base, F, PURPLE_BG, PURPLE)
    cx = x + (W + DX) / 2
    caption(ax, cx, cap_y, "Entrada $\\mathbf{x}$", ["$[1,\\,8,\\,N]$", "8 atributos", "ordem Curva-Z"])
    prev = xe
    x = prev + gap
    for i, (ci, co, k) in enumerate(layers):
        arrow(ax, prev + 0.2, x - 0.2)
        mid = (prev + x) / 2
        ax.text(mid, 1.3, "Conv1d", ha="center", va="bottom", fontsize=11, color=GREEN)
        ax.text(mid, 0.45, "$C=%d$" % k, ha="center", va="bottom", fontsize=10, color=GRAY)
        xe = cuboid(ax, x, base, co, GREEN_BG, GREEN)
        last = i == len(layers) - 1
        p = ci * co * k + co
        cx = x + (W + DX) / 2
        title = "Saída" if last else "Camada oculta %d" % (i + 1)
        if not last:
            xr = xe + 0.05
            xe = cuboid(ax, xr, base, co, ORANGE_BG, ORANGE, w=0.25)
            ax.text(xr + 0.12 + DX / 2, height(co) / 2 + DY + 0.3, "ReLU",
                    ha="center", fontsize=10, color=ORANGE)
        caption(ax, cx, cap_y, title, ["$[1,\\,%d,\\,N]$" % co, "%s parâmetros" % fmt(p)])
        prev = xe
        x = prev + gap

    # divisão em mu / scale
    arrow(ax, prev + 0.2, x - 0.2)
    ax.text((prev + x) / 2, 1.3, "divisão", ha="center", va="bottom", fontsize=11, color=GRAY)
    ax.text((prev + x) / 2, 0.45, "de canais", ha="center", va="bottom", fontsize=10, color=GRAY)
    off = height(F) / 2 + 1.1
    cuboid(ax, x, off, F, GOLD_BG, GOLD)
    cuboid(ax, x, -off, F, GOLD_BG, GOLD)
    cxo = x + (W + DX) / 2
    ax.text(cxo, off + height(F) / 2 + DY + 0.35, "$\\mu$ = out[:, :8]", ha="center", fontsize=11, color=TEXT)
    ax.text(cxo, -off - height(F) / 2 - 0.5, "$\\sigma$ = exp(out[:, 8:]) + $10^{-4}$",
            ha="center", va="top", fontsize=11, color=TEXT)
    ax.text(cxo, -off - height(F) / 2 - 1.8, "$[N,\\,8]$ cada", ha="center", va="top", fontsize=9.5, color=GRAY)
    xend = x + W + DX

    # contêiner tracejado
    pad = 1.8
    x0, x1 = -pad, xend + 5.0
    y0, y1 = ymin - 6.4, height(93) / 2 + DY + 3.4
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fill=False, edgecolor=GREEN, lw=1.6, ls=(0, (5, 3))))
    ax.text(x0 + 0.1, y1 + 0.8, "GaussianARM — arquitetura \"%s\"" % name, ha="left", va="bottom",
            fontsize=16, fontweight="bold", color=GREEN)
    #ax.text(x1, y1 + 0.8, "contexto $C=16$  |  %s parâmetros" % fmt(total), ha="right", va="bottom",
    #        fontsize=12, color=GRAY)
    ax.text((x0 + x1) / 2, y0 + 0.9,
            f"Total de paramêtros: {total}",
            ha="center", va="center", fontsize=9.5, color=GRAY)

    ax.set_xlim(x0 - 0.5, x1 + 0.5)
    ax.set_ylim(y0 - 0.5, y1 + 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    out = "arm_arch_3d_%s.pdf" % name
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Salvo", out)


if __name__ == "__main__":
    for n, l in ARCHS.items():
        draw(n, l)
