"""Generate a synthetic dataset and save it as an ``.npz`` file."""

from __future__ import annotations

import argparse
import numpy as np

from causal_consistency_nn.config import SyntheticDataConfig
from causal_consistency_nn.data.synthetic import generate_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SCM data")
    parser.add_argument(
        "--out", type=str, default="synthetic.npz", help="Output file path"
    )
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--missing-y-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = SyntheticDataConfig(
        n_samples=args.n_samples,
        p=args.p,
        noise=args.noise,
        missing_y_prob=args.missing_y_prob,
        seed=args.seed,
    )
    dataset, (mu0, mu1) = generate_synthetic(cfg)
    X, T, Y = dataset.tensors
    np.savez(
        args.out,
        X=X.numpy(),
        T=T.numpy(),
        Y=Y.numpy(),
        mu0=mu0.numpy(),
        mu1=mu1.numpy(),
    )
    print(f"Saved dataset to {args.out}")


if __name__ == "__main__":  # pragma: no cover - manual script
    main()
