import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianARM(nn.Module):
    def __init__(self, num_features: int, context_size: int = 16):
        super().__init__()
        self.num_features = num_features
        self.context_size = context_size
        
        self.net = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=context_size),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, num_features * 2, kernel_size=1) 
        )
    
    def forward(self, x):
        # x shape: [Batch=1, Features, SeqLen]
        # Pad left by context_size to make it strictly causal
        x_padded = F.pad(x, (self.context_size, 0))
        x_padded = x_padded[..., :-1] 
        
        out = self.net(x_padded)
        out = out.transpose(1, 2).squeeze(0) # [SeqLen, Features*2]
        
        mu = out[:, :self.num_features]
        # Adding a small eps to scale to prevent division by zero
        scale = torch.exp(out[:, self.num_features:]) + 1e-4
        return mu, scale

def get_z_order_indices(x, pos_bits):
    """
    Returns sorting indices based on 2D Morton (Z-order) curve.
    x is assumed to be in [0, 1] range shape (N, 2).
    """
    with torch.no_grad():
        qmax = 2**pos_bits - 1
        x_min, x_max = x.min(dim=0, keepdim=True).values, x.max(dim=0, keepdim=True).values
        scale = torch.clamp((x_max - x_min) / qmax, min=1e-8)
        qx_int = torch.round((x - x_min) / scale).clamp(0, qmax).to(torch.int64)
        
        x_int = qx_int[:, 0]
        y_int = qx_int[:, 1]
        
        morton_code = torch.zeros_like(x_int, dtype=torch.int64)
        for i in range(pos_bits):
            bit_x = (x_int >> i) & 1
            bit_y = (y_int >> i) & 1
            morton_code |= (bit_x << (2 * i))
            morton_code |= (bit_y << (2 * i + 1))
            
        sorted_indices = torch.argsort(morton_code)
    return sorted_indices

def discrete_logistic_prob(mu, scale, symbols, max_symbols):
    """
    Calculates the probability of discrete symbols using a continuous Logistic distribution.
    symbols: [SeqLen, Features]
    max_symbols: [Features] or scalar
    """
    upper_bound = symbols + 0.5
    lower_bound = symbols - 0.5
    
    cdf_upper = torch.sigmoid((upper_bound - mu) / scale)
    cdf_lower = torch.sigmoid((lower_bound - mu) / scale)
    
    prob = cdf_upper - cdf_lower
    # Edge conditions
    prob = torch.where(symbols == 0, cdf_upper, prob)
    # If using tensor for max_symbols, unsqueeze appropriately if passed properly
    # Assuming max_symbol broadcast correctly
    prob = torch.where(symbols == max_symbols, 1.0 - cdf_lower, prob)
    return prob.clamp(min=1e-6)

