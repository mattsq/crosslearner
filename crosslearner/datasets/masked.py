"""Dataset utilities for random feature masking."""

import torch
from torch.utils.data import Dataset


class MaskedFeatureDataset(Dataset):
    """Dataset that returns randomly masked features and the original inputs.

    When the wrapped dataset contains categorical variables ``x_cat`` as the
    second element, the masked dataset yields ``(x_masked, x_cat, x)`` so that
    downstream consumers (e.g. representation pretraining) can pass the
    categorical features through unchanged.
    """

    def __init__(self, dataset: Dataset, mask_prob: float = 0.15) -> None:
        self.dataset = dataset
        self.mask_prob = float(mask_prob)

    def __len__(self) -> int:  # type: ignore
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample = self.dataset[idx]
        x = sample[0]
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        mask = torch.rand_like(x) < self.mask_prob
        x_masked = x.clone()
        x_masked[mask] = 0.0
        # When the underlying dataset provides categorical features as the second
        # element we return them unchanged so the caller can supply them to the
        # model during pretraining.
        if len(sample) == 4:
            x_cat = sample[1]
            if not torch.is_tensor(x_cat):
                x_cat = torch.tensor(x_cat, dtype=torch.long)
            return x_masked, x_cat, x
        return x_masked, x
