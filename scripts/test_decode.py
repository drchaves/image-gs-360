import torch
import torchac

K = 256
sym1 = torch.zeros(100, 1, dtype=torch.int16)
sym2 = torch.ones(100, 1, dtype=torch.int16)

cdf = torch.linspace(0, 1, 257).view(1, 1, 257).repeat(100, 1, 1)

bs1 = torchac.encode_float_cdf(cdf, sym1)
bs2 = torchac.encode_float_cdf(cdf, sym2)

bs_all = bs1 + bs2

# Try to decode bs_all using cdf
out1 = torchac.decode_float_cdf(cdf, bs_all)
print("Decoded sym1 matches?", torch.all(out1 == sym1).item())
