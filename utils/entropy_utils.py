import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClippingSTEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, num_bits=8, clip_val=1.0, stabilize_grads=False):
        # 1. Clipping: limita os valores
        w_clipped = torch.clamp(weight, -clip_val, clip_val)
        
        # 2. Escala e Quantização
        scale = (2**(num_bits - 1) - 1) / clip_val
        q_weight = torch.round(w_clipped * scale)
        
        # 3. Dequantização para o forward
        dq_weight = q_weight / scale
        
        ctx.save_for_backward(weight)
        ctx.clip_val = clip_val
        ctx.stabilize_grads = stabilize_grads
        return dq_weight

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
       
        grad_weight = grad_output.clone()
        if ctx.stabilize_grads:
            # Zera apenas gradientes que empurram o peso ainda mais para fora de [-clip_val, clip_val],
            # permitindo que pesos saturados consigam retornar para a região ativa.
            grad_weight[(weight > ctx.clip_val) & (grad_output < 0)] = 0
            grad_weight[(weight < -ctx.clip_val) & (grad_output > 0)] = 0
        else:
            grad_weight[weight > ctx.clip_val] = 0
            grad_weight[weight < -ctx.clip_val] = 0
        
        return grad_weight, None, None, None

class QuantizedConv1d(nn.Conv1d):
    def __init__(self, *args, num_bits=8, clip_val=1.0, stabilize_grads=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.stabilize_grads = stabilize_grads

    def forward(self, input):
        q_weight = ClippingSTEFunc.apply(self.weight, self.num_bits, self.clip_val, self.stabilize_grads)
        q_bias = ClippingSTEFunc.apply(self.bias, self.num_bits, self.clip_val, self.stabilize_grads) if self.bias is not None else None
        
        return F.conv1d(input, q_weight, q_bias, self.stride,
                        self.padding, self.dilation, self.groups)

class GaussianARM(nn.Module):
    def __init__(self, num_features: int, context_size: int = 16, use_ste: bool = False, arm_bits: int = 8, arch_type: str = "original", stabilize_grads: bool = False,
                 log_scale_min: float = -7.0, log_scale_max: float = 5.0,
                 stab_scale_clamp: bool = True, stab_ste: bool = True):
        # log_scale_min/max: limites do clamp da ativação bruta s_t (ativo somente com stabilize_grads e stab_scale_clamp).
        # stab_scale_clamp / stab_ste: permitem desligar individualmente componentes da estabilização (ablação).
        # Os valores padrão reproduzem exatamente o comportamento anterior.
        super().__init__()
        self.num_features = num_features
        self.context_size = context_size
        self.use_ste = use_ste
        self.arm_bits = arm_bits
        self.arch_type = arch_type
        self.stabilize_grads = stabilize_grads
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.stab_scale_clamp = stab_scale_clamp
        
        Conv = QuantizedConv1d if use_ste else nn.Conv1d
        kwargs = {"num_bits": arm_bits, "clip_val": 1.0, "stabilize_grads": (stabilize_grads and stab_ste)} if use_ste else {}

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
        raw_log_scale = out[:, self.num_features:]
        if self.stabilize_grads and self.stab_scale_clamp:
            # Impede explosão da exponencial (max) e colapso da escala (min)
            raw_log_scale = torch.clamp(raw_log_scale, min=self.log_scale_min, max=self.log_scale_max)
        # Adding a small eps to scale to prevent division by zero
        scale = torch.exp(raw_log_scale) + 1e-4
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


def discrete_logistic_nll(mu, scale, symbols, max_symbols, stabilize_grads=False):
    """
    Calcula a Negative Log-Likelihood em bits (-log2(P)) para a distribuição logística discreta.
    Quando stabilize_grads=True, preserva exatamente o forward e backward onde prob > 1e-6, e
    injeta um gradiente surrogate em log-space (estilo PixelCNN++) apenas nos pontos saturados
    (prob <= 1e-6) onde o clamp zeraria o gradiente, resgatando canais mortos do ARM.
    """
    prob = discrete_logistic_prob(mu, scale, symbols, max_symbols)
    nll_bits = -torch.log2(prob)
    if not stabilize_grads:
        return nll_bits

    # Onde prob > 1e-6, usa o gradiente exato de nll_bits.
    # Onde prob == 1e-6 (saturado no clamp, gradiente = 0), injeta o gradiente assintótico em log-space
    # em relação a mu e scale (com symbols.detach()) para puxar o ARM de volta da região saturada.
    inv_scale = 1.0 / scale
    mid_in_arm = (symbols.detach() - mu) * inv_scale
    log_pdf_mid = mid_in_arm - torch.log(scale) - 2.0 * F.softplus(mid_in_arm)
    tail_surrogate_nll = -log_pdf_mid / math.log(2.0)

    is_saturated = (prob <= 1e-6).detach()
    surrogate_term = torch.where(is_saturated, tail_surrogate_nll - tail_surrogate_nll.detach(), torch.zeros_like(nll_bits))
    return nll_bits + surrogate_term


