import torch
import torchac

K = 256
sym = torch.zeros(1000, 1, dtype=torch.int16)
# If K = 256, symbols are 0..255.
# cdf must have size 257.
cdf = torch.linspace(0, 1, 257).view(1, 1, 257).repeat(1000, 1, 1)

print("CDF shape:", cdf.shape)
out = torchac.encode_float_cdf(cdf, sym)
print("Bytes:", len(out))

