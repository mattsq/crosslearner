import torch
from crosslearner.datasets.toy import get_toy_dataloader
from crosslearner.training.train_acx import train_acx
from crosslearner.evaluation.evaluate import evaluate


def test_train_acx_short():
    torch.manual_seed(0)
    loader, (mu0, mu1) = get_toy_dataloader(batch_size=16, n=64, p=4)
    model = train_acx(loader, p=4, device='cpu', epochs=2)
    X = torch.cat([b[0] for b in loader])
    mu0_all = mu0
    mu1_all = mu1
    metric = evaluate(model, X, mu0_all, mu1_all)
    assert isinstance(metric, float)
    assert metric >= 0.0
