Freezing the Representation Network
===================================

``freeze_phi_epoch`` in :class:`~crosslearner.training.TrainingConfig` stops
updating the shared representation network ``phi`` after a chosen epoch.
Setting ``freeze_phi_epoch`` to an integer freezes all parameters of ``phi``
from that epoch onwards.  Use ``None`` to keep training ``phi`` for the
entire run.

This can be helpful when the encoder should remain fixed while fine-tuning
the outcome or effect heads on a new dataset.

Example
-------

.. code-block:: python

   model_cfg = ModelConfig(p=10)
   train_cfg = TrainingConfig(epochs=5, freeze_phi_epoch=3)
   model = train_acx(loader, model_cfg, train_cfg)

