"""Gera diagramas em PDF das 3 arquiteturas da GaussianARM (original, shallow, deep).

Sem dependências externas (escreve o PDF diretamente), pois matplotlib/TikZ
não estão disponíveis no ambiente. Uso: python3 draw_arm_architectures.py
"""

NUM_FEATURES = 8
CONTEXT = 16

# (c_in, c_out, kernel) por camada Conv1d; ReLU entre todas, exceto após a última.
ARCHS = {
    "original": [(8, 64, 16), (64, 64, 1), (64, 16, 1)],
    "shallow": [(8, 93, 16), (93, 16, 1)],
    "deep": [(8, 53, 16), (53, 53, 1), (53, 53, 1), (53, 16, 1)],
}

COL_IO = (0.85, 0.92, 1.00)
COL_CONV = (1.00, 0.88, 0.70)
COL_RELU = (0.82, 0.95, 0.82)
COL_OUT = (0.93, 0.85, 0.98)
COL_PAD = (0.93, 0.93, 0.93)


class Pdf:
    def __init__(self, w, h):
        self.w, self.h, self.ops = w, h, []

    def _c(self, rgb):
        return "%.3f %.3f %.3f" % rgb

    def box(self, x, y, w, h, fill, lines, dashed=False):
        d = "[4 3] 0 d " if dashed else "[] 0 d "
        self.ops.append("%s rg 0.15 0.15 0.15 RG 1.2 w %s%.1f %.1f %.1f %.1f re B"
                        % (self._c(fill), d, x, y, w, h))
        total = sum(sz * 1.25 for _, sz, _ in lines)
        cy = y + h / 2 + total / 2
        for text, sz, bold in lines:
            cy -= sz * 1.25
            tw = len(text) * sz * (0.56 if bold else 0.52)
            self.text(x + (w - tw) / 2, cy + sz * 0.3, text, sz, bold)

    def text(self, x, y, s, size, bold=False):
        s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.ops.append("BT 0 g /%s %.1f Tf %.1f %.1f Td (%s) Tj ET"
                        % ("F2" if bold else "F1", size, x, y, s))

    def arrow(self, x1, y1, x2, y2):
        self.ops.append("0.15 0.15 0.15 RG 1.4 w [] 0 d %.1f %.1f m %.1f %.1f l S" % (x1, y1, x2 - 5, y2))
        self.ops.append("0.15 0.15 0.15 rg %.1f %.1f m %.1f %.1f l %.1f %.1f l f"
                        % (x2, y2, x2 - 8, y2 + 4, x2 - 8, y2 - 4))

    def save(self, path):
        content = "\n".join(self.ops).encode("cp1252")
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents 4 0 R "
             "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>" % (self.w, self.h)).encode(),
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        ]
        out, offs = bytearray(b"%PDF-1.4\n"), []
        for i, o in enumerate(objs, 1):
            offs.append(len(out))
            out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
        for o in offs:
            out += b"%010d 00000 n \n" % o
        out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
        open(path, "wb").write(bytes(out))


def params(layers):
    return [ci * co * k + co for ci, co, k in layers]


def draw(name, layers):
    nconv = len(layers)
    bw, bh, rw, gap = 130, 100, 62, 34
    n_boxes_w = 2 * 120 + nconv * bw + (nconv - 1) * rw + 190  # in, pad, convs, relus, out
    n_gaps = 2 + (nconv - 1) * 2 + 1 + 1
    W = int(n_boxes_w + n_gaps * gap + 80)
    H = 330
    pdf = Pdf(W, H)
    ps = params(layers)
    total = sum(ps)
    pdf.text(40, H - 40, "GaussianARM - arquitetura \"%s\"" % name, 20, True)
    pdf.text(40, H - 62, "Hiperparâmetros: %d features por gaussiana, contexto K=%d | %d camada(s) oculta(s) | %d parâmetros no total"
             % (NUM_FEATURES, CONTEXT, nconv - 1, total), 11)

    cy = H / 2 - 20
    x = 40
    # Entrada
    pdf.box(x, cy - bh / 2, 120, bh, COL_IO, [("Entrada x", 11, True), ("[1, 8, N]", 11, False),
                                               ("8 atributos da", 9, False), ("gaussiana (Z-order)", 9, False)])
    prev = x + 120
    x = prev + gap
    pdf.arrow(prev, cy, x, cy)
    pdf.box(x, cy - bh / 2, 120, bh, COL_PAD, [("Padding causal", 11, True), ("F.pad(esq, K=16)", 9, False),
                                                ("descarta último", 9, False), ("[1, 8, N+15]", 9, False)], dashed=True)
    prev = x + 120
    for i, ((ci, co, k), p) in enumerate(zip(layers, ps)):
        x = prev + gap
        pdf.arrow(prev, cy, x, cy)
        pdf.box(x, cy - bh / 2, bw, bh, COL_CONV,
                [("Conv1d #%d" % (i + 1), 11, True), ("%d -> %d canais" % (ci, co), 10, False),
                 ("kernel = %d" % k, 10, False), ("%s params" % format(p, ",").replace(",", "."), 9, False)])
        prev = x + bw
        if i < nconv - 1:
            x = prev + gap
            pdf.arrow(prev, cy, x, cy)
            pdf.box(x, cy - 22, rw, 44, COL_RELU, [("ReLU", 11, True)])
            prev = x + rw
    # Saída
    x = prev + gap
    pdf.arrow(prev, cy, x, cy)
    ow = 190
    pdf.box(x, cy + 8, ow, 52, COL_OUT, [("mu = out[:, :8]", 11, True), ("[N, 8] - média prevista", 9, False)])
    pdf.box(x, cy - 60, ow, 52, COL_OUT, [("scale = exp(out[:, 8:]) + 1e-4", 10, True),
                                           ("[N, 8] - escala prevista", 9, False)])
    pdf.text(x, cy + 70, "Saída [1, 16, N] -> transpose", 9)
    pdf.text(40, 52, "Camadas Conv1d viram QuantizedConv1d (STE, arm_bits) quando --use_ste está ativo."
             " Orçamento-alvo: ~13.456 parâmetros (arquitetura original).", 9)
    pdf.text(40, 36, "Parâmetros por camada: " + " + ".join(format(p, ",").replace(",", ".") for p in ps)
             + " = " + format(total, ",").replace(",", "."), 9)
    path = "arm_arch_%s.pdf" % name
    pdf.save(path)
    print("Salvo", path, "(%d params)" % total)


if __name__ == "__main__":
    for n, l in ARCHS.items():
        draw(n, l)

