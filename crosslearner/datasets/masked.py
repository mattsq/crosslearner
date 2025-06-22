import torch
from torch.utils.data import Dataset


class MaskedFeatureDataset(Dataset):
    """Dataset that returns randomly masked features and the original inputs."""

    def __init__(self, dataset: Dataset, mask_prob: float = 0.15) -> None:
        self.dataset = dataset
        self.mask_prob = float(mask_prob)

    def __len__(self) -> int:  # type: ignore
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.dataset[idx][0]
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        mask = torch.rand_like(x) < self.mask_prob
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, x
