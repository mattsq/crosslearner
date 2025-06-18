import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_consistency_nn.train import Settings, train_em
from causal_consistency_nn.config import SyntheticDataConfig


def test_train_end_to_end(tmp_path):
    cfg = SyntheticDataConfig(n_samples=200, p=4, noise=0.1, seed=0)
    settings = Settings(
        data=cfg,
        epochs=3,
        batch_size=32,
        lr=1e-2,
        out_dir=str(tmp_path),
        seed=0,
    )
    losses, _ = train_em(settings)
    assert losses[-1] <= losses[0]
