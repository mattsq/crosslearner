Monitoring Training with TensorBoard
===================================

The ``tensorboard_logdir`` option of :class:`~crosslearner.training.TrainingConfig`
enables detailed logging of the training progress. When set to a directory path
an instance of :class:`torch.utils.tensorboard.SummaryWriter` is created and
standard metrics are written as event files. This includes generator and
adversary losses as well as the validation PEHE or orthogonal risk if either
``val_data`` or ``risk_data`` are provided.

Motivation
----------

Visualising loss curves helps diagnose convergence issues and compare different
settings. TensorBoard provides an interactive dashboard that lets you inspect
how metrics evolve over time. This is useful when tuning hyperparameters or
running longer experiments where printing every epoch would clutter the output.

Usage
-----

Pass a directory to ``tensorboard_logdir`` in the training configuration::

   cfg = TrainingConfig(
       epochs=30,
       tensorboard_logdir="runs/experiment1",
   )
   model = train_acx(loader, ModelConfig(p=10), cfg)

After training launch TensorBoard to view the logs::

   tensorboard --logdir runs/experiment1

When to use it
--------------

Enable ``tensorboard_logdir`` whenever you want a graphical overview of the
training process. It is especially helpful during development to spot unstable
adversarial dynamics or overfitting. For quick tests you can keep it ``None`` to
avoid writing additional files.

References
----------

.. [Abadi2016] Abadi, M., et al. *TensorFlow: A System for Large-Scale Machine
   Learning.* OSDI 2016. Introduces TensorBoard for visualising training
   metrics.
