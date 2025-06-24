import torch
from crosslearner.datasets.random_dag import get_random_dag_dataloader


def test_get_random_dag_dataloader_seed_reproducible():
    _, (mu0_a, mu1_a) = get_random_dag_dataloader(batch_size=2, n=4, p=3, seed=0)
    _, (mu0_b, mu1_b) = get_random_dag_dataloader(batch_size=2, n=4, p=3, seed=0)
    assert torch.allclose(mu0_a, mu0_b)
    assert torch.allclose(mu1_a, mu1_b)
