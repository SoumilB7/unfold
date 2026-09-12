"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations



_SOURCE = """
    import torch
    from torch import nn
    from torch.nn import functional as F

    def select(fallback):
        return fallback

    def kernel(module, q, k, v):
        weights = F.softmax(torch.matmul(q, k), dim=-1)
        return torch.matmul(weights, v)

    class Projection:
        def __init__(self, config):
            self.q = nn.Linear(config.hidden, config.hidden)
            self.k = nn.Linear(config.hidden, config.hidden)
            self.v = nn.Linear(config.hidden, config.hidden)

        def forward(self, hidden, context=None, cache=None):
            source = context if context is not None else hidden
            q = self.q(hidden)
            k = self.k(source)
            v = self.v(source)
            operation = select(kernel)
            return operation(self, q, k, v)

    class Cell:
        def __init__(self, config):
            self.first = Projection(config)
            self.second = Projection(config)

        def forward(self, hidden, context=None, cache=None):
            hidden = self.first(hidden, cache=cache)
            if context is not None:
                hidden = hidden + self.second(
                    hidden, context=context, cache=cache)
            return hidden

    class Core:
        def __init__(self, config):
            self.cells = nn.ModuleList(
                [Cell(config) for _ in range(config.layers)])
        def forward(self, hidden, context=None):
            for cell in self.cells:
                hidden = cell(hidden, context=context)
            return hidden

    class Shell:
        base_model_prefix = "core"
        def __init__(self, config):
            self.core = Core(config)
"""
