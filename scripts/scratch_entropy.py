import torch

def discretize_parameter(x: torch.Tensor, num_bits: int):
    """
    Simulates quantization and returns the discrete integer symbols 
    along with min and max necessary for dequantization.
    """
    qmin, qmax = 0, 2**num_bits - 1
    min_val, max_val = x.min().item(), x.max().item()
    if max_val == min_val:
        scale = 1e-8
    else:
        scale = max((max_val - min_val) / (qmax - qmin), 1e-8)
    
    # [0, 2^num_bits - 1]
    q_x_int = torch.round((x - min_val) / scale).clamp(qmin, qmax).to(torch.int32)
    return q_x_int, min_val, max_val, scale

def get_morton_code_2d(x_int: torch.Tensor, y_int: torch.Tensor, num_bits: int):
    """
    Interleaves bits of x_int and y_int to create a Morton code representation.
    """
    # Ensure they are 64-bit int to prevent overflow when shifting up to 64 bits total
    morton_code = torch.zeros_like(x_int, dtype=torch.int64)
    
    for i in range(num_bits):
        bit_x = (x_int >> i) & 1
        bit_y = (y_int >> i) & 1
        
        # Interleave: typically y is most significant per pair, x is least
        # 1 bit of x at position 2*i, 1 bit of y at position 2*i + 1
        morton_code |= (bit_x.to(torch.int64) << (2 * i))
        morton_code |= (bit_y.to(torch.int64) << (2 * i + 1))
        
    return morton_code

def test_pipeline():
    print("Iniciando teste de Discretização e Ordenação Morton...")
    num_gaussians = 5
    pos_bits = 8 # Exemplo usando precisão de 8 bits para a posição
    
    # 1. Simulamos as posições XY de 5 Gaussianas espalhadas
    # Valores aleatórios de [0, 1] no float32
    torch.manual_seed(42)
    xy = torch.rand(num_gaussians, 2, dtype=torch.float32)
    
    x = xy[:, 0]
    y = xy[:, 1]
    
    # 2. Descompor em inteiros
    qx, min_x, max_x, scale_x = discretize_parameter(x, pos_bits)
    qy, min_y, max_y, scale_y = discretize_parameter(y, pos_bits)
    
    print("\n--- Posições XY (Original Float32) ---")
    print(xy)
    
    print("\n--- Símbolos Quantizados (qx, qy) ---")
    print(torch.stack([qx, qy], dim=1))
    
    # 3. Gerar Códigos de Morton (Z-order)
    morton_codes = get_morton_code_2d(qx, qy, pos_bits)
    print("\n--- Códigos de Morton ---")
    print(morton_codes)
    
    # 4. Ordenação e recuperação de Índices
    sorted_indices = torch.argsort(morton_codes)
    print("\n--- Índices Ordenados ---")
    print(sorted_indices)
    
    # 5. Aplica a ordenação na nossa nuvem de gaussianas quantizada
    qx_sorted = qx[sorted_indices]
    qy_sorted = qy[sorted_indices]
    print("\n--- Posições (qx, qy) Ordenadas em Curva Z ---")
    print(torch.stack([qx_sorted, qy_sorted], dim=1))

if __name__ == "__main__":
    test_pipeline()
