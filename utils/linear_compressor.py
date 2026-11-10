"""
Compressão linear paralela em blocos Z-order (Morton) para o Image-GS.

Modo `--compression_mode linear_subspace`:
    1. Ordena as gaussianas pela curva de Morton (a partir de mu).
    2. Agrupa em M = ceil(N / B) blocos de B gaussianas (padding = repete a última gaussiana).
    3. Posições  : ancora por bloco (centroide, float16) + deslocamentos locais quantizados
                   uniformemente com faixa max-abs do bloco (float16).
    4. Atributos : (1/s, theta, cor) padronizados globalmente (media/desvio em float16) e projetados
                   num subespaco ortogonal de K componentes (SVD/KLT ao longo do bloco, base V_K em float16).
    5. Coeficientes Z quantizados com tamanho fixo (`quant_bits`) e empacotados em bits (bit-packing).

Nao ha codificacao de entropia: o bloco `m` ocupa sempre o mesmo numero de bits, portanto
seu endereco no stream e O(1) (`m * D * K * quant_bits`), permitindo decodificacao paralela/aleatoria por tile.
A descompressao e 100% vetorizada (unpack de bits + torch.bmm), sem loops por gaussiana.

Interface:
    comp = LinearBlockCompressor(block_size=16, num_components=6, quant_bits=8)
    bitstream = comp.compress({"xy": [N,2], "scale": [N,2], "rot": [N,1], "feat": [N,C]})
    gaussians = comp.decompress(bitstream)      # mesmas chaves, em ordem de Morton
    bits      = comp.compute_bits(bitstream)    # contabilidade exata
"""
import math

import torch

from utils.entropy_utils import get_z_order_indices

_HEADER_KEYS = ("meta", "mean", "std", "V", "zmin", "zmax")
_ANCHOR_KEYS = ("anchors", "pos_range")
_POS_KEYS = ("pos_q",)
_COEF_KEYS = ("coef_q",)


def pack_bits(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Empacota inteiros nao-negativos `q` (qualquer shape) com `bits` bits cada em um uint8 1D."""
    q = q.reshape(-1)
    if bits == 8:
        return q.to(torch.uint8)
    shifts = torch.arange(bits, device=q.device, dtype=torch.int64)
    bit_mat = ((q.to(torch.int64)[:, None] >> shifts) & 1).to(torch.uint8).reshape(-1)
    pad = (-bit_mat.numel()) % 8
    if pad:
        bit_mat = torch.cat([bit_mat, bit_mat.new_zeros(pad)])
    weights = (2 ** torch.arange(8, device=q.device)).to(torch.int32)
    return (bit_mat.view(-1, 8).to(torch.int32) * weights).sum(dim=1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, n: int, bits: int) -> torch.Tensor:
    """Inverso de `pack_bits`: retorna `n` inteiros (int64) de `bits` bits cada."""
    if bits == 8:
        return packed[:n].to(torch.int64)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32)
    bit_mat = ((packed.to(torch.int32)[:, None] >> shifts) & 1).reshape(-1)[: n * bits]
    weights = (2 ** torch.arange(bits, device=packed.device)).to(torch.int64)
    return (bit_mat.view(n, bits).to(torch.int64) * weights).sum(dim=1)


def _tensor_bits(t: torch.Tensor) -> int:
    return t.numel() * t.element_size() * 8


class LinearBlockCompressor:
    def __init__(self, block_size: int = 16, num_components: int = 6, quant_bits: int = 8,
                 pos_quant_bits: int = 0, morton_bits: int = 16):
        if block_size < 1:
            raise ValueError("block_size deve ser >= 1")
        if not 1 <= quant_bits <= 16:
            raise ValueError("quant_bits deve estar entre 1 e 16")
        pos_quant_bits = quant_bits if pos_quant_bits in (0, None) else pos_quant_bits
        if not 1 <= pos_quant_bits <= 16:
            raise ValueError("pos_quant_bits deve estar entre 1 e 16")
        self.block_size = int(block_size)
        self.num_components = max(1, min(int(num_components), self.block_size))
        self.quant_bits = int(quant_bits)
        self.pos_quant_bits = int(pos_quant_bits)
        self.morton_bits = int(morton_bits)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _z_range(zmin16: torch.Tensor, zmax16: torch.Tensor):
        """Faixa de quantizacao calculada de forma identica no encoder e no decoder (a partir do float16)."""
        lo = zmin16.float()
        rng = (zmax16.float() - lo).clamp_min(1e-4)
        return lo, rng

    @staticmethod
    def _pos_range(r16: torch.Tensor) -> torch.Tensor:
        return r16.float().clamp_min(1e-6)

    # ----------------------------------------------------------------- compress
    @torch.no_grad()
    def compress(self, gaussians: dict) -> dict:
        xy = gaussians["xy"].detach().float()
        attrs = torch.cat([gaussians["scale"].detach().float(),
                           gaussians["rot"].detach().float(),
                           gaussians["feat"].detach().float()], dim=-1)  # [N, D]
        N, D = attrs.shape
        B = self.block_size
        M = math.ceil(N / B)
        pad = M * B - N

        # 1) Ordem de Morton (a ordem nao precisa ser armazenada: a soma de gaussianas e invariante a permutacao)
        order = get_z_order_indices(xy, self.morton_bits)
        xy, attrs = xy[order], attrs[order]

        # 2) Padding de borda repetindo a ultima gaussiana (sem zeros artificiais)
        if pad > 0:
            xy = torch.cat([xy, xy[-1:].expand(pad, -1)], dim=0)
            attrs = torch.cat([attrs, attrs[-1:].expand(pad, -1)], dim=0)

        # 3) Posicoes: ancora (centroide) por bloco em float16 + deslocamentos locais
        xy_b = xy.view(M, B, 2)
        anchors16 = xy_b.mean(dim=1).half()                                   # [M, 2]
        delta = xy_b - anchors16.float()[:, None, :]                          # [M, B, 2]
        r16 = (delta.abs().amax(dim=1) * (1.0 + 1e-3)).half()                 # [M, 2] faixa max-abs
        r = self._pos_range(r16)
        pmax = 2 ** self.pos_quant_bits - 1
        u = (delta / r[:, None, :]).clamp(-1.0, 1.0)
        pos_q = torch.round((u + 1.0) * 0.5 * pmax).clamp(0, pmax)            # [M, B, 2]

        # 4) Padronizacao global (estatisticas sobre as N gaussianas reais, salvas em float16)
        mean16 = attrs[:N].mean(dim=0).half()
        std16 = attrs[:N].std(dim=0, unbiased=False).clamp_min(1e-4).half()
        Xn = (attrs - mean16.float()) / std16.float()                         # [M*B, D]
        X = Xn.view(M, B, D).permute(2, 0, 1).contiguous()                    # [D, M, B]

        # 5) Subespaco linear (SVD/KLT) por canal ao longo do bloco
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)                   # [D, min(M,B), B]
        K = min(self.num_components, B, Vh.shape[1])
        V16 = Vh[:, :K, :].transpose(1, 2).contiguous().half()                # [D, B, K]
        Z = torch.bmm(X, V16.float())                                         # [D, M, K]

        # 6) Quantizacao uniforme de tamanho fixo (faixa global por (canal, componente))
        zmin16, zmax16 = Z.amin(dim=1).half(), Z.amax(dim=1).half()           # [D, K]
        lo, rng = self._z_range(zmin16, zmax16)
        qmax = 2 ** self.quant_bits - 1
        coef_q = torch.round((Z - lo[:, None, :]) / rng[:, None, :] * qmax).clamp(0, qmax)
        coef_q = coef_q.permute(1, 0, 2).contiguous()                         # [M, D, K]: bloco contiguo

        return {
            "meta": torch.tensor([N, B, K, self.quant_bits, self.pos_quant_bits], dtype=torch.int32, device=xy.device),
            "mean": mean16, "std": std16, "V": V16, "zmin": zmin16, "zmax": zmax16,
            "anchors": anchors16, "pos_range": r16,
            "pos_q": pack_bits(pos_q, self.pos_quant_bits),
            "coef_q": pack_bits(coef_q, self.quant_bits),
        }

    # --------------------------------------------------------------- decompress
    @torch.no_grad()
    def decompress(self, bitstream: dict) -> dict:
        N, B, K, qb, pb = [int(v) for v in bitstream["meta"].tolist()]
        mean, std = bitstream["mean"].float(), bitstream["std"].float()
        V = bitstream["V"].float()                                            # [D, B, K]
        D = mean.numel()
        M = math.ceil(N / B)

        # Coeficientes: unpack + dequantizacao + reconstrucao em lote (todos os M blocos de uma vez)
        coef_q = unpack_bits(bitstream["coef_q"], M * D * K, qb).view(M, D, K).permute(1, 0, 2).float()
        lo, rng = self._z_range(bitstream["zmin"], bitstream["zmax"])
        Z = coef_q / (2 ** qb - 1) * rng[:, None, :] + lo[:, None, :]         # [D, M, K]
        X = torch.bmm(Z, V.transpose(1, 2))                                   # [D, M, B]
        attrs = X * std[:, None, None] + mean[:, None, None]
        attrs = attrs.permute(1, 2, 0).reshape(M * B, D)[:N]                  # [N, D]

        # Posicoes: ancora + deslocamento dequantizado
        pos_q = unpack_bits(bitstream["pos_q"], M * B * 2, pb).view(M, B, 2).float()
        r = self._pos_range(bitstream["pos_range"])
        delta = (pos_q / (2 ** pb - 1) * 2.0 - 1.0) * r[:, None, :]
        xy = (bitstream["anchors"].float()[:, None, :] + delta).reshape(M * B, 2)[:N]

        # Dominios validos. theta e simetrico em pi para elipses, logo `remainder` nao altera a renderizacao.
        scale = attrs[:, 0:2].clamp_min(1e-6)
        rot = torch.remainder(attrs[:, 2:3], math.pi).clamp(0.0, math.pi)
        feat = attrs[:, 3:]
        return {"xy": xy.clamp(0.0, 1.0), "scale": scale, "rot": rot, "feat": feat}

    # --------------------------------------------------------- bitrate accounting
    @staticmethod
    def compute_bits(bitstream: dict) -> dict:
        """Tamanho exato (em bits) dos tensores armazenados, separado por categoria."""
        header = sum(_tensor_bits(bitstream[k]) for k in _HEADER_KEYS)
        anchors = sum(_tensor_bits(bitstream[k]) for k in _ANCHOR_KEYS)
        pos = sum(_tensor_bits(bitstream[k]) for k in _POS_KEYS)
        coef = sum(_tensor_bits(bitstream[k]) for k in _COEF_KEYS)
        return {"header": header, "anchors": anchors, "pos_deltas": pos, "coefficients": coef,
                "total": header + anchors + pos + coef}

    def estimate_bits(self, num_gaussians: int, dim: int) -> int:
        """Total de bits previsto (formula fechada, identica a `compute_bits` do stream real)."""
        B = self.block_size
        M = math.ceil(num_gaussians / B)
        K = min(self.num_components, B, M)
        header = 5 * 32 + (2 * dim + dim * B * K + 2 * dim * K) * 16
        anchors = M * 2 * 2 * 16
        pos = math.ceil(M * B * 2 * self.pos_quant_bits / 8) * 8
        coef = math.ceil(M * dim * K * self.quant_bits / 8) * 8
        return header + anchors + pos + coef

    @classmethod
    def bpp(cls, bitstream: dict, num_pixels: int) -> float:
        return cls.compute_bits(bitstream)["total"] / float(num_pixels)
