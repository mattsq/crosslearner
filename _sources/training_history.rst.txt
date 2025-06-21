Tracking Training History
========================

The ``return_history`` option of :class:`~crosslearner.training.TrainingConfig`
determines whether :func:`~crosslearner.training.train_acx` returns a
``History`` object alongside the trained model.  ``History`` is a list of
:class:`~crosslearner.training.EpochStats` dataclasses recording losses,
gradient norms and learning rates for each epoch.

Motivation
----------

Analysing how losses evolve can reveal mode collapse or overfitting that is not
obvious from the final metric alone.  Access to the full history allows custom
visualisation beyond what TensorBoard provides.

Usage
-----

Set ``return_history=True`` when calling the training routine::

   cfg = TrainingConfig(
       epochs=30,
       return_history=True,
   )
   model, history = train_acx(loader, ModelConfig(p=10), cfg)

Pass ``history`` to :func:`crosslearner.visualization.plot_losses` or
implement your own analytics.

When to use it
--------------

Enable ``return_history`` during experimentation or when you need to log metrics
for offline analysis.  Disable it in production training loops to avoid the
additional return value and keep the interface simple.

References
----------

.. [Abadi2016] Abadi, M., et al. *TensorFlow: A System for Large-Scale Machine
   Learning.* OSDI 2016. Discusses tracking and visualising training metrics.
