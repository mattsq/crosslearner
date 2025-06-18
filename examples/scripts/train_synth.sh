#!/bin/bash
# Train causal_consistency_nn on a synthetic dataset
python -m causal_consistency_nn.train --out-dir "$1"
