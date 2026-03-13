"""Causal Perceptron Network used for NSA mock generation in this repository.

This module provides the minimal API used by `make_mocks.py`:
- `NoiseDistribution`
- `CausalPerceptronNetwork`

The implementation is intentionally focused on the paper use case:
continuous variables with additive exogenous noise as an explicit input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class NoiseDistribution:
    """Simple exogenous noise sampler used by CPN nodes."""

    distribution_type: str = "beta"
    alpha: float = 2.0
    beta: float = 5.0
    mean: float = 0.0
    std: float = 1.0
    low: float = 0.0
    high: float = 1.0

    def __init__(self, distribution_type: str = "beta", **kwargs: Any) -> None:
        object.__setattr__(self, "distribution_type", distribution_type)
        object.__setattr__(self, "alpha", float(kwargs.get("alpha", 2.0)))
        object.__setattr__(self, "beta", float(kwargs.get("beta", 5.0)))
        object.__setattr__(self, "mean", float(kwargs.get("mean", 0.0)))
        object.__setattr__(self, "std", float(kwargs.get("std", 1.0)))
        object.__setattr__(self, "low", float(kwargs.get("low", 0.0)))
        object.__setattr__(self, "high", float(kwargs.get("high", 1.0)))

    def sample(self, size: int) -> np.ndarray:
        if self.distribution_type == "beta":
            return np.random.beta(self.alpha, self.beta, size)
        if self.distribution_type == "normal":
            return np.random.normal(self.mean, self.std, size)
        if self.distribution_type == "uniform":
            return np.random.uniform(self.low, self.high, size)
        raise ValueError(f"Unsupported distribution type: {self.distribution_type}")


class MultiLayerPerceptron(nn.Module):
    """Node-local MLP mapping parent values + noise to one continuous output."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        activation: nn.Module,
        input_scale: float,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if input_scale <= 0:
            raise ValueError("input_scale must be > 0")

        self.input_scale = float(input_scale)

        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_layers:
            if width < 1:
                raise ValueError("All hidden layer sizes must be >= 1")
            linear = nn.Linear(prev, width)
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            layers.append(linear)
            layers.append(activation if isinstance(activation, nn.Module) else nn.ReLU())
            prev = width

        out = nn.Linear(prev, 1)
        nn.init.kaiming_uniform_(out.weight, nonlinearity="linear")
        layers.append(out)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * self.input_scale
        x_norm = (x_scaled - x_scaled.mean(dim=0, keepdim=True)) / (
            x_scaled.std(dim=0, keepdim=True) + 1e-8
        )
        return self.layers(x_norm)


class CausalPerceptronNetwork:
    """Generate synthetic data by traversing a Tetrad DAG in valid causal order."""

    def __init__(
        self,
        graph,
        num_samples: int,
        noise_distributions: dict,
        rescale_min: float | None = None,
        rescale_max: float | None = None,
        hidden_dimensions: list[int] | None = None,
        input_scale: float = 1.0,
        activation_module: nn.Module = nn.ReLU(),
        nonlinearity: str = "relu",
        parallelize: bool = True,
        device: str = "cpu",
        discrete_prob: float = 0.0,
        min_num_categories: int = 2,
        max_num_categories: int = 5,
        seed: int | None = None,
    ) -> None:
        del nonlinearity, min_num_categories, max_num_categories

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not self.is_acyclic(graph):
            raise ValueError("Graph contains directed cycles")
        if discrete_prob not in (0, 0.0):
            raise ValueError(
                "Paper-focused CPN supports continuous variables only; use discrete_prob=0"
            )

        self.graph = graph
        self.num_samples = int(num_samples)
        self.noise_distributions = noise_distributions
        self.rescale_min = rescale_min
        self.rescale_max = rescale_max
        self.hidden_dimensions = hidden_dimensions if hidden_dimensions is not None else [10]
        self.input_scale = float(input_scale)
        self.activation_module = activation_module
        self.parallelize = bool(parallelize)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        nodes = list(self.graph.getNodes())
        missing_noise = [node for node in nodes if node not in self.noise_distributions]
        if missing_noise:
            raise ValueError("noise_distributions must contain an entry for each graph node")
        for node in nodes:
            if not isinstance(self.noise_distributions[node], NoiseDistribution):
                raise ValueError("All noise_distributions values must be NoiseDistribution instances")

        self.node_to_mlp: dict[Any, nn.Module] = {}
        for node in nodes:
            input_dim = len(list(graph.getParents(node))) + 1
            mlp = MultiLayerPerceptron(
                input_dim=input_dim,
                hidden_layers=self.hidden_dimensions,
                activation=self.activation_module,
                input_scale=self.input_scale,
            )
            if self.parallelize and torch.cuda.device_count() > 1:
                mlp = nn.DataParallel(mlp)
            self.node_to_mlp[node] = mlp.to(self.device).to(torch.float64)

    def generate_data(self) -> pd.DataFrame:
        nodes = list(self.graph.getNodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        order = self.topological_sort(self.graph)

        data = torch.zeros((self.num_samples, len(nodes)), device=self.device, dtype=torch.float64)

        for node in order:
            parents = list(self.graph.getParents(node))
            parent_idx = [node_to_idx[p] for p in parents]
            parent_data = data[:, parent_idx] if parent_idx else torch.empty(
                (self.num_samples, 0), device=self.device, dtype=torch.float64
            )

            noise_np = self.noise_distributions[node].sample(self.num_samples)
            noise = torch.tensor(noise_np, dtype=torch.float64, device=self.device).unsqueeze(1)
            inputs = torch.cat((parent_data, noise), dim=1)

            node_values = self.node_to_mlp[node](inputs).squeeze(1)
            if self.rescale_min is not None and self.rescale_max is not None:
                node_values = self.rescale_data(node_values)
            data[:, node_to_idx[node]] = node_values

        data_np = data.detach().cpu().numpy().astype(np.float64)
        col_names = [str(n) for n in nodes]
        return pd.DataFrame(data_np, columns=col_names)

    def rescale_data(self, values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        vmin = torch.min(values)
        vmax = torch.max(values)
        if float(vmax - vmin) <= eps:
            return values
        return self.rescale_min + (values - vmin) * (self.rescale_max - self.rescale_min) / (vmax - vmin)

    @staticmethod
    def is_acyclic(graph) -> bool:
        return not graph.paths().existsDirectedCycle()

    @staticmethod
    def topological_sort(graph):
        return graph.paths().getValidOrder(graph.getNodes(), True)
