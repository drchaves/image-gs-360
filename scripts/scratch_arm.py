import torch
import torch.nn as nn
import torch.nn.functional as F
import torchac

class GaussianARM(nn.Module):
    def __init__(self, num_features: int, context_size: int = 16):
        super().__init__()
        self.num_features = num_features
        # Causal Conv1D: looking at past `context_size` elements
        self.context_size = context_size
        
        # We will use padded causal convolution
        self.net = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=context_size),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, num_features * 2, kernel_size=1) 
            # Outputs mean and log_scale for each feature
        )
    
    def forward(self, x):
        # x shape: [Batch=1, Features, SeqLen]
        # Pad left by context_size to make it strictly causal
        # so the output at position i only sees positions up to i-1
        x_padded = F.pad(x, (self.context_size, 0))
        # strictly causal shift: cut the last element of padding and don't see current element
        x_padded = x_padded[..., :-1] 
        
        out = self.net(x_padded)
        # out shape: [1, Features*2, SeqLen]
        out = out.transpose(1, 2).squeeze(0) # [SeqLen, Features*2]
        
        mu = out[:, :self.num_features]
        # We use log_scale so scale is always positive
        scale = torch.exp(out[:, self.num_features:]) 
        return mu, scale

def get_cdf(mu, scale, max_symbol=255):
    """
    Computes the CDF of a Logistic(mu, scale) evaluated at symbol boundaries.
    """
    # Create bins for CDF
    symbols = torch.arange(max_symbol + 1, dtype=torch.float32, device=mu.device)
    # Shape of symbols: [max_symbol + 1] -> broadcast with mu, scale
    
    # We evaluate CDF at symbol - 0.5 for the lower bound.
    # To get a CDF format for torchac:
    # cd[0] = 0
    # cdf[1] = P(X <= 0)
    # ...
    # cdf[max_symbol] = P(X <= max_symbol-1)
    # cdf[max_symbol + 1] = 1
    
    # Let's expand dimensions: shape will be [SeqLen, Features, max_symbol + 2]
    # symbols shape [1, 1, max_symbol + 1]
    b_symbols = symbols.view(1, 1, -1) - 0.5
    b_mu = mu.unsqueeze(-1)
    b_scale = scale.unsqueeze(-1)
    
    logistic_cdf = torch.sigmoid((b_symbols - b_mu) / (b_scale + 1e-6))
    
    # Pad with 0 at the beginning and 1 at the end to satisfy torchac requirements
    zeros = torch.zeros_like(logistic_cdf[..., :1])
    ones = torch.ones_like(logistic_cdf[..., :1])
    
    # Combine: 0, F(0-0.5), F(1-0.5), ..., F(V-0.5)
    # Wait, torchac expects:
    # cdf[..., 0] = 0.0
    # cdf[..., -1] = 1.0 (approximated/clamped by the library, but exact 1.0 is required in CDF end)
    cdf = torch.cat([zeros, logistic_cdf, ones], dim=-1)
    
    # Clamp to ensure strict monotonically increasing bounds in FP32
    cdf = torch.clamp(cdf, 0.0, 1.0)
    return cdf

def test_arm_and_torchac():
    print("Testing Causal ARM + TorchAC...")
    
    seq_len = 100
    num_features = 2 # e.g., X and Y position
    
    # 1. Simulating Z-ordered discrete symbols (our Gaussians)
    # They usually have correlation, so we'll simulate a random correlated walk
    symbols = torch.zeros((seq_len, num_features), dtype=torch.int32)
    symbols[0] = torch.tensor([128, 128])
    for i in range(1, seq_len):
        step = torch.randint(-5, 6, (num_features,))
        symbols[i] = torch.clamp(symbols[i-1] + step, 0, 255)
        
    print(f"Símbolos gerados: {seq_len} elementos")
    
    # Normalize features to float for the ARM input
    x_float = (symbols.float() / 255.0).t().unsqueeze(0) # [1, 2, 100]
    
    # 2. Forward pass no ARM
    arm = GaussianARM(num_features, context_size=8)
    mu_norm, scale_norm = arm(x_float)
    
    # Denormalize mu to [0, 255], scale accordingly
    mu = mu_norm * 255.0
    scale = scale_norm * 255.0
    
    # 3. Gerar a CDF (Distribuição Logística)
    # A precisão (max_symbol) é 255, gerando CDF de tamanho 257
    cdf = get_cdf(mu, scale, max_symbol=255) # Forma: [100, 2, 257]
    cdf = cdf.to(torch.float32) # torchac strictly needs float32 CPU
    
    # 4. Encoding
    # torchac.encode_float_cdf requires sym to be [B, ..., N], same shape as CDF minus the last dim
    # cdf is [100, 2, 257], symbols is [100, 2]
    # Let's ensure types
    symbols = symbols.to(torch.int16) 
    
    # Encodar!
    try:
        # byte_stream é a string binária comprimida!
        byte_stream = torchac.encode_float_cdf(cdf, symbols)
        print(f"Compressão bem-sucedida! Tamanho original: {symbols.numel() * 8} bits (não comprimido de 8 bits).")
        print(f"Tamanho comprimido com ARM não otimizado: {len(byte_stream) * 8} bits.")
    except Exception as e:
        print("Erro no encode:", e)

if __name__ == "__main__":
    test_arm_and_torchac()
