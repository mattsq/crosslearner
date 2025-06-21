ACX Architecture
================

The AC-X model implements an adversarially trained version of the
``X``-learner. It consists of four main components:

``phi``
  Shared representation network that maps the covariates to a hidden
  feature space of dimension ``rep_dim``.
``mu0`` and ``mu1``
  Outcome heads predicting control and treated responses from the
  representation.
``tau``
  Treatment‑effect head estimating the conditional average treatment
  effect (CATE).
``disc``
  Discriminator used for adversarial training.

All networks are configurable multi-layer perceptrons created by
:class:`crosslearner.models.MLP`. Residual connections and dropout can be
turned on or off for each module individually.

Configuration options
---------------------

The :class:`crosslearner.training.ModelConfig` dataclass exposes the
following parameters:

``p``
  Number of covariates.
``rep_dim``
  Size of the shared representation.
``phi_layers``
  Hidden layer sizes for ``phi``.
``head_layers``
  Hidden layers for the outcome and effect heads.
``disc_layers``
  Hidden layers for the discriminator.
``activation``
  Activation function used in all MLPs.
``phi_dropout``, ``head_dropout``, ``disc_dropout``
  Dropout probabilities for each component.
``residual`` and ``*_residual``
  Enable residual connections globally or per component.
``disc_pack``
  Number of samples concatenated for the discriminator.
``weight_init``
  Initialisation scheme applied to linear layers. Options include
  ``"xavier_uniform"`` and ``"kaiming_uniform"``.
``batch_norm``
  If ``True`` insert ``BatchNorm1d`` layers after each hidden linear
  layer.

New features
------------

Recent versions add optional weight initialisation and batch
normalisation. When enabled, every linear layer is initialised according
to the chosen scheme and followed by a batch-normalisation layer. This
can improve stability when training on more complex datasets. A
contrastive representation loss can also be enabled via
``contrastive_weight`` to further balance covariates across treatment
groups.

An optional adaptive regularization scheme ``adaptive_reg`` can tune the
gradient penalty strength on the fly based on the discriminator loss to
improve adversarial stability.

Representations can further be disentangled into confounder-, outcome- and
instrument-specific parts by setting ``disentangle=True`` and providing sizes
for ``rep_dim_c``, ``rep_dim_a`` and ``rep_dim_i``. Two auxiliary adversaries
controlled via ``adv_t_weight`` and ``adv_y_weight`` then encourage the desired
independence structure.

Customising parameters
----------------------

The features can be controlled either directly when instantiating
:class:`crosslearner.models.ACX` or through
:class:`~crosslearner.training.ModelConfig`::

    from crosslearner.models import ACX
    from crosslearner.training import ModelConfig

    # Direct instantiation
    model = ACX(
        p=10,
        rep_dim=128,
        weight_init="kaiming_uniform",
        batch_norm=True,
    )

    # Using a configuration object
    cfg = ModelConfig(
        p=10,
        rep_dim=128,
        weight_init="kaiming_uniform",
        batch_norm=True,
    )
    model = ACX(
        cfg.p,
        rep_dim=cfg.rep_dim,
        phi_layers=cfg.phi_layers,
        head_layers=cfg.head_layers,
        disc_layers=cfg.disc_layers,
        activation=cfg.activation,
        phi_dropout=cfg.phi_dropout,
        head_dropout=cfg.head_dropout,
        disc_dropout=cfg.disc_dropout,
        residual=cfg.residual,
        phi_residual=cfg.phi_residual,
        head_residual=cfg.head_residual,
        disc_residual=cfg.disc_residual,
        disc_pack=cfg.disc_pack,
        weight_init=cfg.weight_init,
        batch_norm=cfg.batch_norm,
    )

References
----------

.. [Kueng2018] Kueng, R., et al. *Neural Network Methods for Causal Inference: A
   Review of DragonNet and Related Approaches.* 2018. Describes the DragonNet
   architecture that inspires AC-X.
.. [Künzel2019] Künzel, S., Sekhon, J., Bickel, P., & Yu, B. *Metalearners for
   Estimating Heterogeneous Treatment Effects using Machine Learning.* PNAS
   2019. Introduces the X-learner strategy.
