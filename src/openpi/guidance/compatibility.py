"""Torch observation/action compatibility model, matching exported egg archives."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional


class Tower(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 64))

    def forward(self, x):
        return functional.normalize(self.net(x), dim=-1, eps=1e-8)


class Guide(nn.Module):
    def __init__(self, *, no_touch=False):
        super().__init__()
        self.obs = Tower(509)
        self.act = Tower(848)
        self.no_touch = no_touch

    def observation(self, o):
        if self.no_touch:
            o = torch.cat([torch.zeros_like(o[:, :480]), o[:, 480:]], -1)
        return self.obs(o)

    def action(self, a, t):
        f = torch.tensor([1.0, 2.0, 4.0, 8.0], device=a.device) * np.pi
        phase = t[:, None] * f
        return self.act(torch.cat([a.flatten(1), phase.sin(), phase.cos()], -1))

    def score(self, o, a, t):
        return (self.observation(o) * self.action(a, t)).sum(-1)
