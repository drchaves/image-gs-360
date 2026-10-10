import torch
import torch.nn as nn
import torch.nn.functional as F

class ClippingSTEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, num_bits=8, clip_val=1.0):
        # 1. Clipping: limita os valores
        w_clipped = torch.clamp(weight, -clip_val, clip_val)
        
        # 2. Escala e Quantização
        scale = (2**(num_bits - 1) - 1) / clip_val
        q_weight = torch.round(w_clipped * scale)
        
        # 3. Dequantização para o forward
        dq_weight = q_weight / scale
        
        ctx.save_for_backward(weight)
        ctx.clip_val = clip_val
        return dq_weight

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
       
        grad_weight = grad_output.clone()
        grad_weight[weight > ctx.clip_val] = 0
        grad_weight[weight < -ctx.clip_val] = 0
        
        return grad_weight, None, None

class QuantizedConv1d(nn.Conv1d):
    def __init__(self, *args, num_bits=8, clip_val=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.clip_val = clip_val

    def forward(self, input):
        q_weight = ClippingSTEFunc.apply(self.weight, self.num_bits, self.clip_val)
        q_bias = ClippingSTEFunc.apply(self.bias, self.num_bits, self.clip_val) if self.bias is not None else None
        
        return F.conv1d(input, q_weight, q_bias, self.stride,
                        self.padding, self.dilation, self.groups)

class GaussianARM(nn.Module):
    def __init__(self, num_features: int, context_size: int = 16, use_ste: bool = False, arm_bits: int = 8, arch_type: str = "original"):
        super().__init__()
        self.num_features = num_features
        self.context_size = context_size
        self.use_ste = use_ste
        self.arm_bits = arm_bits
        self.arch_type = arch_type
        
        Conv = QuantizedConv1d if use_ste else nn.Conv1d
        kwargs = {"num_bits": arm_bits, "clip_val": 1.0} if use_ste else {}

        if arch_type == "original":
            # Parâmetros: ~13.456 (assumindo 8 features e context_size=16)
            self.net = nn.Sequential(
                Conv(num_features, 64, kernel_size=context_size, **kwargs),
                nn.ReLU(),
                Conv(64, 64, kernel_size=1, **kwargs),
                nn.ReLU(),
                Conv(64, num_features * 2, kernel_size=1, **kwargs) 
            )
        elif arch_type == "shallow":
            # Mais rasa e mais larga (W=93) -> Parâmetros: ~13.501
            self.net = nn.Sequential(
                Conv(num_features, 93, kernel_size=context_size, **kwargs),
                nn.ReLU(),
                Conv(93, num_features * 2, kernel_size=1, **kwargs) 
            )
        elif arch_type == "deep":
            # Mais profunda e mais estreita (W=53) -> Parâmetros: ~13.425
            self.net = nn.Sequential(
                Conv(num_features, 53, kernel_size=context_size, **kwargs),
                nn.ReLU(),
                Conv(53, 53, kernel_size=1, **kwargs),
                nn.ReLU(),
                Conv(53, 53, kernel_size=1, **kwargs),
                nn.ReLU(),
                Conv(53, num_features * 2, kernel_size=1, **kwargs) 
            )
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")
    
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
        # Grid dimension
        qmax = 2**pos_bits - 1

        # Normalize x to [0, 1]
        x_min, x_max = x.min(dim=0, keepdim=True).values, x.max(dim=0, keepdim=True).values

        # Calculate scale to map the range to [0, qmax]
        # -- tamanho físico de cada quadradinho
        scale = torch.clamp((x_max - x_min) / qmax, min=1e-8)

        # Convert to integer grid coordinates
        qx_int = torch.round((x - x_min) / scale).clamp(0, qmax).to(torch.int64)
        
        # Get x and y coordinates
        x_int = qx_int[:, 0]
        y_int = qx_int[:, 1]
        
        # Interleave bits of x and y to create Morton code
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

    # 10 -> [9.5 - 10.5], 20 -> [19.5 - 20.5] etc
    upper_bound = symbols + 0.5
    lower_bound = symbols - 0.5
    
    # Cumulative distribution function (CDF)
    cdf_upper = torch.sigmoid((upper_bound - mu) / scale)
    cdf_lower = torch.sigmoid((lower_bound - mu) / scale)
    
    prob = cdf_upper - cdf_lower
    # Edge conditions
    prob = torch.where(symbols == 0, cdf_upper, prob)
    prob = torch.where(symbols == max_symbols, 1.0 - cdf_lower, prob)

    # Prevent division by zero
    return prob.clamp(min=1e-6)

