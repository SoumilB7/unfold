
import torch
from torch import nn
from torch.nn import functional as F

def eager(module, query, key, value, cap=None):
    weights = torch.matmul(query, key.transpose(-1, -2))
    weights = weights + 0
    weights = F.softmax(weights, dim=-1)
    return torch.matmul(weights, value)

class Mixer:
    def __init__(self, config):
        self.config = config
        self.limit = self.config.attention_cap
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        return eager(self, query, key, value, cap=self.limit)


class Cell:
    def __init__(self, config):
        self.mixer = Mixer(config)
    def forward(self, x):
        return self.mixer(x)

class Core:
    def __init__(self, config):
        self.items = nn.ModuleList([Cell(config) for _ in range(config.layers)])
    def forward(self, x):
        for item in self.items:
            x = item(x)
        return x

class Wrapper:
    base_model_prefix = "core"
    def __init__(self, config):
        self.core = Core(config)
