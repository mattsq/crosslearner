import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple multi-layer perceptron used throughout the framework."""

    def __init__(self, in_dim: int, out_dim: int, hidden=(128, 64)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ACX(nn.Module):
    """Adversarial-Consistency X-learner model."""

    def __init__(self, p: int, rep_dim: int = 64):
        super().__init__()
        self.phi = MLP(p, rep_dim, hidden=(128,))
        self.mu0 = MLP(rep_dim, 1, hidden=(64,))
        self.mu1 = MLP(rep_dim, 1, hidden=(64,))
        self.tau = MLP(rep_dim, 1, hidden=(64,))
        self.disc = MLP(rep_dim + 2, 1, hidden=(64,))

    def forward(self, x):
        h = self.phi(x)
        m0 = self.mu0(h)
        m1 = self.mu1(h)
        tau = self.tau(h)
        return h, m0, m1, tau

    def discriminator(self, h, y, t):
        return self.disc(torch.cat([h, y, t], dim=1))
