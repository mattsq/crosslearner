Custom Optimizer Arguments
==========================

The ``opt_g_kwargs`` and ``opt_d_kwargs`` fields of
:class:`~crosslearner.training.TrainingConfig` allow you to pass additional
keyword arguments to the generator and discriminator optimisers.

Motivation
----------

While the default ``Adam`` optimiser works well for most experiments, you may
want to tweak its momentum parameters or use completely different optimisers.
The configuration entries ``opt_g_kwargs`` and ``opt_d_kwargs`` let you
specify such custom options without modifying the training loop.

Usage
-----

Set ``optimizer`` to either a string name (``"adam"``, ``"sgd"``, ``"adamw"`` or
``"rmsprop"``) or directly to an optimiser class.  Extra keyword arguments for
these optimisers can then be provided via ``opt_g_kwargs`` and
``opt_d_kwargs``::

   cfg = TrainingConfig(
       epochs=30,
       optimizer="adam",
       opt_g_kwargs={"betas": (0.5, 0.999)},
       opt_d_kwargs={"betas": (0.9, 0.999)},
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

When to use it
--------------

Use these options whenever you need fine-grained control over optimisation
hyperparameters.  For instance, you can experiment with different momentum
values, enable Nesterov momentum for ``SGD`` or supply weight decay settings
separately for the generator and discriminator.  Leaving the dictionaries empty
retains the defaults of the selected optimiser.

References
----------

.. [Kingma2014] Kingma, D., & Ba, J. *Adam: A Method for Stochastic
   Optimization.* ICLR 2015.
