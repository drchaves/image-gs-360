"""Execução do benchmark CLIC 2025 (30 imagens de teste ~2K em media/clic2025_image_test) comparando:
  1. Baseline: Image-GS original (14 bits, sem ARM) em N in {10k, 30k, 50k, 100k}
  2. Nosso método vencedor: GaussianARM (Shallow, C=2, stabilize_grads=True, STE 8 bits)
     em N in {10k, 30k, 50k, 100k} para 3 valores de lambda:
       - 0.0001 (melhor resultado geral)
       - 0.0005 (imediatamente inferior)
       - 0.0075 (pior / mais agressivo)

Este script compartilha exatamente a implementação de run_clic2025.py.
"""
from run_clic2025 import main

if __name__ == "__main__":
    main()
