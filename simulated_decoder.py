import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath('gsplat'))

# Imports from the project
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
    return f"{args.log_root}/{args.exp_name}/{folder}"

def decode_and_render(ckpt_path, output_path):
    print(f"=== Decodificador Simulado ===")
    print(f"Carregando o 'Bitstream' (via Checkpoint): {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    num_gaussians = checkpoint['state_dict']['xy'].shape[0]
    
    parser = argparse.ArgumentParser()
    parser = load_cfg(cfg_path="cfgs/default.yaml", parser=parser)
    args, _ = parser.parse_known_args([])
    
    args.eval = True
    args.ckpt_file = ckpt_path
    args.num_gaussians = num_gaussians
    args.log_dir = get_log_dir(args)
    
    print("Inicializando o Rasterizador...")
    ImageGS = GaussianSplatting2D(args)
    
    print(f"Total de Gaussianos Recuperados: {ImageGS.num_gaussians}")
    print("-----------------------------------")
    print("Parâmetros brutos extraídos (Prontos para render):")
    print(f"- Posições (X,Y): {ImageGS.xy.shape}")
    print(f"- Escalas: {ImageGS.scale.shape}")
    print(f"- Rotações: {ImageGS.rot.shape}")
    print(f"- Cores (Features): {ImageGS.feat.shape}")
    print("-----------------------------------")
    
    print(f"Renderizando a imagem final para a resolução original ({ImageGS.img_h}x{ImageGS.img_w})...")
    images = ImageGS._render_images(upsample=False)
    
    ImageGS._separate_and_save_images(images=images, channels=ImageGS.input_channels, path=output_path)
    print(f"Sucesso! Imagem reconstruída salva em: {output_path}")

if __name__ == "__main__":
    import glob
    
    ckpts = glob.glob("results/**/checkpoints/*.pt", recursive=True)
    if not ckpts:
        print("Nenhum arquivo .pt encontrado. Rode o treinamento primeiro.")
        sys.exit(1)
        
    # Pega o último arquivo gerado
    ckpts.sort(key=os.path.getmtime)
    ckpt_path = os.path.abspath(ckpts[-1])
    
    output_path = "results/imagem_decodificada"
    decode_and_render(ckpt_path, output_path)
