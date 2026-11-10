import argparse

import os
import sys
sys.path.insert(0, os.path.abspath('gsplat'))

import numpy as np
import torch

from model import GaussianSplatting2D
from utils.misc_utils import load_cfg


def get_gaussian_cfg(args):
    gaussian_cfg = f"num-{args.num_gaussians:d}"
    if args.disable_inverse_scale:
        gaussian_cfg += f"_scale-{args.init_scale:.1f}"
    else:
        gaussian_cfg += f"_inv-scale-{args.init_scale:.1f}"
    if not args.quantize:
        args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits = 32, 32, 32, 32
    min_bits = min(args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits)
    max_bits = max(args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits)
    if min_bits < 4 or max_bits > 32:
        raise ValueError(
            f"Bit precision must be between 4 and 32 but got: {args.pos_bits:d}, {args.scale_bits:d}, {args.rot_bits:d}, {args.feat_bits:d}")
    gaussian_cfg += f"_bits-{args.pos_bits:d}-{args.scale_bits:d}-{args.rot_bits:d}-{args.feat_bits:d}"
    if not args.disable_topk_norm:
        gaussian_cfg += f"_top-{args.topk:d}"
    gaussian_cfg += f"_{args.init_mode[0]}-{args.init_random_ratio:.1f}"
    return gaussian_cfg


def get_log_dir(args):
    gaussian_cfg = get_gaussian_cfg(args)
    loss_cfg = f"l1-{args.l1_loss_ratio:.1f}_l2-{args.l2_loss_ratio:.1f}_ssim-{args.ssim_loss_ratio:.1f}"
    folder = f"{gaussian_cfg}_{loss_cfg}"
    if args.downsample:
        folder += f"_ds-{args.downsample_ratio:.1f}"
    if not args.disable_lr_schedule:
        folder += f"_decay-{args.max_decay_times:d}-{args.decay_ratio:.1f}"
    if not args.disable_prog_optim:
        folder += "_prog"
    if getattr(args, 'compression_mode', 'autoregressive') == 'linear_subspace':
        folder += f"_lin-B{args.block_size:d}-K{args.num_components:d}-Q{args.quant_bits:d}"
    return f"{args.log_root}/{args.exp_name}/{folder}"


def main(args):
    args.log_dir = get_log_dir(args)
    ImageGS = GaussianSplatting2D(args)
    if args.eval:
        ImageGS.render(render_height=args.render_height)
    else:
        ImageGS.optimize()


if __name__ == "__main__":
    torch.hub.set_dir("models/torch")
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ste', action='store_true', help='Habilita a quantização dos parâmetros do ARM via STE para reduzir o BPP da rede.')
    parser.add_argument('--arm_bits', type=int, default=8, help='Quantidade de bits para a quantização STE do ARM (padrão: 8).')
    parser.add_argument('--arm_arch', type=str, default="original", choices=["original", "shallow", "deep"], help='Arquitetura do ARM.')
    parser.add_argument('--arm_context', type=int, default=16, help='Tamanho da janela de contexto causal (C) do ARM (padrão: 16).')
    parser.add_argument('--early_stopping', action='store_true', help='Ativa a parada antecipada quando o PSNR atinge um platô.')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Paciência para o early stopping (num. de avaliações sem melhora).')
    parser.add_argument('--stabilize_grads', action='store_true', help='Ativa estabilização contra vanishing/exploding gradients (log-space NLL, LeakyReLU, STE e scale clamp, clip_grad_norm e preservação do estado do Adam).')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Norma máxima para o gradient clipping quando --stabilize_grads está ativo (padrão: 1.0).')
    parser.add_argument('--log_scale_min', type=float, default=-7.0, help='Limite inferior do clamp da ativação bruta de escala s_t (ativo com --stabilize_grads; padrão: -7).')
    parser.add_argument('--log_scale_max', type=float, default=5.0, help='Limite superior do clamp da ativação bruta de escala s_t (ativo com --stabilize_grads; padrão: 5).')
    parser.add_argument('--stab_disable', type=str, nargs='*', default=[], choices=['surrogate', 'clamp', 'clip', 'ste', 'adam'], help='Ablação: desliga componentes individuais da estabilização (surrogate=gradiente substituto nas caudas, clamp=limite de escala, clip=recorte de norma do ARM, ste=STE com retorno à faixa ativa, adam=preservação dos momentos).')
    parser.add_argument('--compression_mode', type=str, default="autoregressive", choices=["none", "autoregressive", "linear_subspace"], help='Modo de compressão: none (sem ARM), autoregressive (ARM + codificação aritmética, comportamento original) ou linear_subspace (PCA/SVD em blocos de Morton com quantização de tamanho fixo).')
    parser.add_argument('--block_size', type=int, default=16, help='[linear_subspace] Número de gaussianas por bloco de Morton (B).')
    parser.add_argument('--num_components', type=int, default=6, help='[linear_subspace] Componentes principais retidos por bloco (K <= B).')
    parser.add_argument('--quant_bits', type=int, default=8, help='[linear_subspace] Bits por coeficiente projetado (1-16).')
    parser.add_argument('--pos_quant_bits', type=int, default=0, help='[linear_subspace] Bits por deslocamento local de posição (0 = usa --quant_bits).')
    parser = load_cfg(cfg_path="cfgs/default.yaml", parser=parser)
    arguments = parser.parse_args()
    main(arguments)
