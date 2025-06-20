Spectral Normalisation for Stable Training
=========================================

The ``spectral_norm`` option wraps every linear layer of the model in
``torch.nn.utils.spectral_norm``. This constrains each layer's spectral
norm—its largest singular value—to one, effectively bounding the
network's Lipschitz constant. By restricting how sharply the discriminator can
change with respect to its inputs, spectral normalisation often leads to more
stable adversarial training.

Motivation
----------

Gradient penalties such as WGAN-GP enforce a Lipschitz constraint by
optimising an additional loss term. Spectral normalisation achieves a similar
constraint directly on the weight matrices without extra gradient
computations. It can prevent the discriminator from dominating the generator
and is particularly useful when the discriminator has many layers or when
training on small datasets.

Usage
-----

Enable spectral normalisation by passing ``spectral_norm=True`` to
:class:`~crosslearner.training.TrainingConfig`::

   cfg = TrainingConfig(
       epochs=30,
       spectral_norm=True,
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

Internally, :func:`~crosslearner.utils.apply_spectral_norm` traverses the
model and applies ``torch.nn.utils.spectral_norm`` to all ``nn.Linear``
modules. If an exponential moving average (``ema_decay``) model is used, the
same wrapping is applied to the EMA copy as well.

When to use it
--------------

Use ``spectral_norm`` when you observe unstable discriminator behaviour or
when gradient penalties slow down training. It pairs well with other
stabilisation techniques such as feature matching and gradient reversal.
Avoid combining it with explicit weight clipping, as both attempt to limit
the discriminator's capacity. If training becomes sluggish, consider lowering
``lambda_gp`` or disabling gradient penalties altogether when spectral
normalisation is enabled.
