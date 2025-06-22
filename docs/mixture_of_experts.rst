Mixture-of-Experts Potential-Outcome Heads
=========================================

Setting ``moe_experts`` greater than ``1`` in :class:`~crosslearner.training.ModelConfig`
replaces the single potential-outcome heads with a set of expert pairs. A gating
network ``g(x)`` softly selects experts for each sample and the final predictions
are the weighted sum of their outputs.

This allows the model to specialise to heterogeneous sub-populations while
sharing a common representation. To encourage sparse selections, the trainer adds
an entropy penalty on the gating distribution controlled by ``moe_entropy_weight``
from :class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       moe_entropy_weight=0.1,
   )
   model_cfg = ModelConfig(p=10, moe_experts=4)
   model = train_acx(loader, model_cfg, cfg)

Existing adversarial and consistency losses operate on the aggregated outputs so
no other changes are required.
