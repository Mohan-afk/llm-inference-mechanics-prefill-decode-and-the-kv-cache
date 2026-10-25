"""
LLM Inference Mechanics: Prefill, Decode and the KV Cache

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - RMSNorm
import torch
import torch.nn as nn
import torch.nn.functional as F

def rms_norm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x*torch.rsqrt(variance+eps)*weight

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        # TODO: weight parameter of ones
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)

def swiglu(x, w1, w3, w2):
    return w2(F.silu(w1(x)) * w3(x))

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        # TODO: w1, w3 (d -> hidden), w2 (hidden -> d), no bias
        self.w1 = nn.Linear(d, hidden, bias = False)
        self.w3 = nn.Linear(d, hidden, bias = False)
        self.w2 = nn.Linear(hidden, d, bias = False)

    def forward(self, x):
        return swiglu(x, self.w1, self.w3, self.w2)

