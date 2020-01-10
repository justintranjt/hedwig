import numpy as np
import torch

dropout = 0.3
sz = 300 + 1024

rnd = torch.rand((sz))
print(rnd.shape, rnd)

x = rnd.bernoulli_(1-dropout)

print(torch.sum(x) / sz )

# mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)

print(x)
