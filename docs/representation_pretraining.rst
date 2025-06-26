Representation Pretraining
==========================

``pretrain_epochs`` enables an unsupervised warm-up phase where the encoder
is trained to reconstruct masked features.  During this stage the model is fed
corrupted inputs from :class:`crosslearner.datasets.MaskedFeatureDataset` and a
linear decoder tries to predict the original features.  Only the shared
representation network ``phi`` and this decoder are updated.

When ``verbose`` is enabled the average reconstruction loss for each
pretraining epoch is printed to the console.

``pretrain_mask_prob`` controls the fraction of features set to zero, mimicking
missing covariates.  ``pretrain_lr`` can override the encoder's learning rate
for this phase while ``finetune_lr`` specifies the learning rate for subsequent
training.  When ``finetune_lr`` is not provided ``lr_g`` is reduced by a factor
of ``0.1`` after pretraining.

This initialization can stabilise adversarial training on small datasets or
when the encoder has many parameters.

Example usage::

   cfg = TrainingConfig(
       pretrain_epochs=5,
       pretrain_mask_prob=0.2,
       epochs=30,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)
