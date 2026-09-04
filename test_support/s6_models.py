"""Importable synthetic modules used by the S6 subprocess poisons."""
from __future__ import annotations

import socket
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F


class InventoryFixture(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = int(config.get("width", 4))
        self.layers = nn.ModuleList([nn.Linear(self.width, self.width) for _ in range(2)])
        self.embed = nn.Embedding(8, self.width)
        self.head = nn.Linear(self.width, 8, bias=False)
        self.head.weight = self.embed.weight
        self.optional = nn.Linear(self.width, self.width) if config.get("optional") else None
        self.unconditional_none = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class MutatingConfigModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.value = config["nested"].pop("value")


class NetworkModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        socket.create_connection(("example.com", 80), timeout=1)


class SubprocessNetworkModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        subprocess.run(["curl", "https://example.com"], check=True)


class SlowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        time.sleep(float(config.get("seconds", 10)))


class NoisyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("x" * int(config.get("characters", 1024 * 1024)))


class MemoryModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        total = int(config.get("bytes", 2 * 1024**3))
        chunk = int(config.get("chunk_bytes", 64 * 1024**2))
        self.payload = []
        while sum(map(len, self.payload)) < total:
            block = bytearray(min(chunk, total - sum(map(len, self.payload))))
            for offset in range(0, len(block), 4096):
                block[offset] = 1
            self.payload.append(block)


class LazyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cache = None

    def forward(self, x):
        if self.cache is None:
            self.cache = nn.Identity()
        return self.cache(x + 1) * 2


class FunctionalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(int(config.get("width", 4)))

    def forward(self, x):
        q, k = torch.chunk(x, 2, dim=-1)
        merged = torch.cat((q, k), dim=-1)
        merged = F.layer_norm(merged, merged.shape[-1:])
        merged = F.silu(merged) + F.gelu(merged)
        return merged * self.norm(x)


class DataDependentModel(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        if x.sum().item() > 0:
            return x + 1
        return x * 2


class TupleInputModel(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, pair, nested):
        return pair[0] + pair[1] + nested["value"]
