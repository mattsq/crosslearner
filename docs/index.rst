crosslearner documentation
==========================

This manual collects theoretical background, usage examples and the complete
API reference for the AC‑X implementation. Follow the links below to learn about
the training procedure, hyperparameter sweeps and available modules.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   usage_examples
   fitting_acx_model
   datasets

.. toctree::
   :maxdepth: 2
   :caption: Architecture and Theory

   acx_architecture
   theory

.. toctree::
   :maxdepth: 2
   :caption: Training Configuration

   hyperparameter_sweeps
   optimizer_kwargs
   warm_start
   ttur
   risk_early_stopping
   training_history
   tensorboard_logging
   adaptive_batch_size

.. toctree::
   :maxdepth: 2
   :caption: Regularization and Tricks

   gradient_reversal
   feature_matching
   label_smoothing
   spectral_norm
   exponential_moving_average
   weight_clipping
   wgan_gp
   r1_r2_regularization
   instance_noise
   contrastive_loss
   mmd_regularization
   consistency_regularization
   discriminator_augmentation
   active_counterfactual_augmentation
   unrolled_discriminator
   mixture_of_experts
   tau_head_bias
   representation_pretraining
   representation_disentanglement
   doubly_robust
   uncertainty



.. toctree::
   :maxdepth: 2
   :caption: API

   api/modules
