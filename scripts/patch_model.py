import re

with open('model.py', 'r') as f:
    content = f.read()

# 1. Add imports
imports = """
from utils.quantization_utils import ste_quantize
from utils.entropy_utils import GaussianARM, get_z_order_indices, discrete_logistic_prob
"""
content = content.replace("from utils.quantization_utils import ste_quantize", imports)

# 2. Add ARM and rate_loss_ratio init
init_loss_old = "    def _init_loss(self, args):"
init_loss_new = """    def _init_loss(self, args):
        self.rate_loss_ratio = getattr(args, 'rate_loss_ratio', 1e-4) # Entropy loss
        self.rate_loss = None
        if self.rate_loss_ratio > 0:
            num_f = 2 + 2 + 1 + self.feat_dim # xy, scale, rot, feat
            self.arm = GaussianARM(num_features=num_f).to(self.device)
"""
content = content.replace(init_loss_old, init_loss_new)

# 3. Add ARM to optimizer in _init_optimization
opt_old = "self.optimizer = torch.optim.Adam([{'params': self.xy, 'lr': self.pos_lr},"
opt_new = """params = [{'params': self.xy, 'lr': self.pos_lr},
                          {'params': self.scale, 'lr': self.scale_lr},
                          {'params': self.rot, 'lr': self.rot_lr},
                          {'params': self.feat, 'lr': self.feat_lr}]
        if getattr(self, 'rate_loss_ratio', 0) > 0 and hasattr(self, 'arm'):
            params.append({'params': self.arm.parameters(), 'lr': 1e-3})
        self.optimizer = torch.optim.Adam(params)
        # """
# We have two places where optimizer is defined. We need to replace them carefully.
content = re.sub(r"self\.optimizer = torch\.optim\.Adam\(\[\{'params': self\.xy.*?\n.*?\{'params': self\.feat, 'lr': self\.feat_lr\}\]\)", opt_new.strip(), content, flags=re.DOTALL)


# 4. Modify _get_total_loss to add ARM rate loss
get_loss_old = "def _get_total_loss(self, images):"
get_loss_new = """def _get_total_loss(self, images):
        self.total_loss = 0
        self.rate_loss = None
        if getattr(self, 'rate_loss_ratio', 0) > 0 and hasattr(self, 'arm'):
            # Convert to symbols (0 to 2^B-1)
            qmax_p = 2**self.pos_bits - 1
            x_min, x_max = self.xy.min(dim=0, keepdim=True).values, self.xy.min(dim=0, keepdim=True).values
            # Fast approx scaling (for demo purposes)
            sym_xy = ((self.xy - self.xy.min()) / (self.xy.max() - self.xy.min() + 1e-8) * qmax_p).clamp(0, qmax_p)
            
            qmax_s = 2**self.scale_bits - 1
            sym_scale = ((self.scale - self.scale.min()) / (self.scale.max() - self.scale.min() + 1e-8) * qmax_s).clamp(0, qmax_s)
            
            qmax_r = 2**self.rot_bits - 1
            sym_rot = ((self.rot - self.rot.min()) / (self.rot.max() - self.rot.min() + 1e-8) * qmax_r).clamp(0, qmax_r)
            
            qmax_f = 2**self.feat_bits - 1
            sym_feat = ((self.feat - self.feat.min()) / (self.feat.max() - self.feat.min() + 1e-8) * qmax_f).clamp(0, qmax_f)
            
            symbols = torch.cat([sym_xy, sym_scale, sym_rot, sym_feat], dim=-1).to(torch.float32)
            
            # Sort via Z-order (Morton) based on spatial coords
            with torch.no_grad():
                indices = get_z_order_indices(sym_xy, self.pos_bits)
                
            sorted_symbols = symbols[indices]
            
            # Forward pass no ARM
            # Shape expects [1, Features, SeqLen]
            arm_in = (sorted_symbols / 255.0).t().unsqueeze(0) 
            mu, scale = self.arm(arm_in)
            
            # Calculate negative log likelihood (rate loss)
            prob = discrete_logistic_prob(mu, scale, sorted_symbols, qmax_p) # approx max
            nll = -torch.log2(prob).mean()
            self.rate_loss = nll * self.rate_loss_ratio
            self.total_loss += self.rate_loss

"""
# Remove the total_loss = 0 initialization since we put it in the replacement
content = content.replace("    def _get_total_loss(self, images):\n        self.total_loss = 0", get_loss_new)

# Fix double inclusion if script reruns
if "get_z_order_indices" not in open('model.py').read():
    with open('model.py', 'w') as f:
        f.write(content)
else:
    print("Already patched")
