import torch
from crosslearner.evaluation.metrics import pehe
from crosslearner.models.acx import ACX


def evaluate(model: ACX, X: torch.Tensor, mu0: torch.Tensor, mu1: torch.Tensor):
    model.eval()
    with torch.no_grad():
        _, _, _, tau_hat = model(X)
    tau_true = mu1 - mu0
    return pehe(tau_hat, tau_true)
