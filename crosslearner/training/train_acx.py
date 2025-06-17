import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Iterable, Optional, Tuple, Type
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from crosslearner.training.history import EpochStats, History
from crosslearner.evaluation.evaluate import evaluate
from crosslearner.utils import set_seed
from crosslearner.training.config import ModelConfig, TrainingConfig

from crosslearner.models.acx import ACX, _get_activation
from crosslearner.training.grl import grad_reverse


def _make_regressor(inp: int, hid: Iterable[int] = (64, 64)) -> nn.Sequential:
    """Return a simple fully connected regressor.

    Args:
        inp: Number of input features.
        hid: Sizes of the hidden layers.

    Returns:
        Sequential network ending with a single linear output.
    """

    layers: list[nn.Module] = []
    d = inp
    for h in hid:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _make_propensity_net(inp: int, hid: Iterable[int] = (64, 64)) -> nn.Sequential:
    """Return a sigmoid-activated regressor for propensity scores."""

    net = _make_regressor(inp, hid)
    net.add_module("sigmoid", nn.Sigmoid())
    return net


def _estimate_nuisances(
    X: torch.Tensor,
    T: torch.Tensor,
    Y: torch.Tensor,
    *,
    folds: int = 5,
    lr: float = 1e-3,
    batch: int = 256,
    propensity_epochs: int = 500,
    outcome_epochs: int = 3,
    early_stop: int = 10,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cross-fitted nuisance predictions.

    Args:
        X: Covariates ``(n, p)``.
        T: Treatment indicators ``(n, 1)``.
        Y: Observed outcomes ``(n, 1)``.
        folds: Number of cross-fitting folds.
        lr: Learning rate used for the nuisance regressors.
        batch: Mini-batch size for the outcome regressors.
        device: Device used for training the nuisances.
        seed: Random seed for reproducibility.

    Returns:
        Tuple ``(e_hat, mu0_hat, mu1_hat)`` with cross-fitted propensity and
        outcome predictions.
    """

    bce = nn.BCELoss()
    mse = nn.MSELoss()

    set_seed(seed)
    kfold = StratifiedKFold(folds, shuffle=True, random_state=seed)
    e_hat = torch.empty_like(T, device=device)
    mu0_hat = torch.empty_like(Y, device=device)
    mu1_hat = torch.empty_like(Y, device=device)

    for train_idx, val_idx in kfold.split(X.cpu(), T.cpu()):
        Xtr, Ttr, Ytr = X[train_idx], T[train_idx], Y[train_idx]
        Xva = X[val_idx]

        # split fold again for early stopping
        n = Xtr.shape[0]
        split = int(0.8 * n)
        X_train, X_val = Xtr[:split], Xtr[split:]
        T_train, T_val = Ttr[:split], Ttr[split:]
        Y_train, Y_val = Ytr[:split], Ytr[split:]

        prop = _make_propensity_net(X.shape[1]).to(device)
        opt_p = torch.optim.Adam(prop.parameters(), lr)
        best_state = {k: v.detach().clone() for k, v in prop.state_dict().items()}
        best_loss = float("inf")
        no_improve = 0
        for _ in range(propensity_epochs):
            pred = prop(X_train)
            loss = bce(pred, T_train)
            opt_p.zero_grad()
            loss.backward()
            opt_p.step()
            val_loss = bce(prop(X_val), T_val).item()
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in prop.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop:
                    break
        prop.load_state_dict(best_state)
        e_hat[val_idx] = prop(Xva).detach()

        mu0 = _make_regressor(X.shape[1]).to(device)
        mu1 = _make_regressor(X.shape[1]).to(device)
        opt_mu = torch.optim.Adam(list(mu0.parameters()) + list(mu1.parameters()), lr)
        ds = torch.utils.data.TensorDataset(X_train, T_train, Y_train)
        val_ds = torch.utils.data.TensorDataset(X_val, T_val, Y_val)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
        best_mu0 = {k: v.detach().clone() for k, v in mu0.state_dict().items()}
        best_mu1 = {k: v.detach().clone() for k, v in mu1.state_dict().items()}
        best_val = float("inf")
        no_improve = 0
        for _ in range(outcome_epochs):
            for xb, tb, yb in loader:
                pred0, pred1 = mu0(xb), mu1(xb)
                loss = torch.tensor(0.0, device=device)
                mask0 = tb == 0
                mask1 = tb == 1
                if mask0.any():
                    loss = loss + mse(pred0[mask0], yb[mask0])
                if mask1.any():
                    loss = loss + mse(pred1[mask1], yb[mask1])
                opt_mu.zero_grad()
                loss.backward()
                opt_mu.step()
            with torch.no_grad():
                val_loss = 0.0
                count = 0
                for xb, tb, yb in torch.utils.data.DataLoader(val_ds, batch_size=batch):
                    pred0, pred1 = mu0(xb), mu1(xb)
                    mask0 = tb == 0
                    mask1 = tb == 1
                    loss = torch.tensor(0.0, device=device)
                    if mask0.any():
                        loss = loss + mse(pred0[mask0], yb[mask0])
                    if mask1.any():
                        loss = loss + mse(pred1[mask1], yb[mask1])
                    val_loss += loss.item()
                    count += 1
                val_loss /= max(count, 1)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_mu0 = {
                        k: v.detach().clone() for k, v in mu0.state_dict().items()
                    }
                    best_mu1 = {
                        k: v.detach().clone() for k, v in mu1.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= early_stop:
                        break
        mu0.load_state_dict(best_mu0)
        mu1.load_state_dict(best_mu1)
        mu0_hat[val_idx] = mu0(Xva).detach()
        mu1_hat[val_idx] = mu1(Xva).detach()

    return e_hat, mu0_hat, mu1_hat


@torch.no_grad()
def _orthogonal_risk(
    tau_hat: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    e_hat: torch.Tensor,
    mu0_hat: torch.Tensor,
    mu1_hat: torch.Tensor,
) -> float:
    """Return the orthogonal risk.

    Args:
        tau_hat: Predicted treatment effects.
        y: Observed outcomes ``(n, 1)``.
        t: Treatment indicators ``(n, 1)``.
        e_hat: Cross-fitted propensity estimates.
        mu0_hat: Predicted outcome under control.
        mu1_hat: Predicted outcome under treatment.

    Returns:
        Scalar orthogonal risk value.
    """

    mse = nn.MSELoss()
    y_resid = y - torch.where(t.bool(), mu1_hat, mu0_hat)
    loss = mse(y_resid - (t - e_hat) * tau_hat, torch.zeros_like(y))
    return loss.item()


def train_acx(
    loader: DataLoader,
    p: int,
    *,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    rep_dim: int = 64,
    phi_layers: Iterable[int] | None = (128,),
    head_layers: Iterable[int] | None = (64,),
    disc_layers: Iterable[int] | None = (64,),
    activation: str | Callable[[], nn.Module] = "relu",
    phi_dropout: float = 0.0,
    head_dropout: float = 0.0,
    disc_dropout: float = 0.0,
    residual: bool = False,
    device: Optional[str] = None,
    seed: int | None = None,
    epochs: int = 30,
    alpha_out: float = 1.0,
    beta_cons: float = 10.0,
    gamma_adv: float = 1.0,
    lr_g: float = 1e-3,
    lr_d: float = 1e-3,
    optimizer: str | Type[torch.optim.Optimizer] = "adam",
    opt_g_kwargs: dict | None = None,
    opt_d_kwargs: dict | None = None,
    lr_scheduler: str | Type[torch.optim.lr_scheduler._LRScheduler] | None = None,
    sched_g_kwargs: dict | None = None,
    sched_d_kwargs: dict | None = None,
    grad_clip: float = 2.0,
    warm_start: int = 0,
    use_wgan_gp: bool = False,
    spectral_norm: bool = False,
    feature_matching: bool = False,
    label_smoothing: bool = False,
    instance_noise: bool = False,
    gradient_reversal: bool = False,
    ttur: bool = False,
    lambda_gp: float = 10.0,
    eta_fm: float = 5.0,
    grl_weight: float = 1.0,
    tensorboard_logdir: Optional[str] = None,
    weight_clip: Optional[float] = None,
    val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    risk_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    risk_folds: int = 5,
    nuisance_propensity_epochs: int = 500,
    nuisance_outcome_epochs: int = 3,
    nuisance_early_stop: int = 10,
    patience: int = 0,
    verbose: bool = True,
    return_history: bool = False,
) -> ACX | tuple[ACX, History]:
    """Train AC-X model with optional GAN tricks.

    Args:
        loader: PyTorch dataloader yielding ``(X, T, Y)`` batches.
        p: Number of covariates.
        model_config: Optional :class:`ModelConfig` with architecture
            hyperparameters. When supplied, the corresponding keyword arguments
            are ignored.
        training_config: Optional :class:`TrainingConfig` holding optimisation
            parameters. When given, the individual keyword arguments are
            overridden.
        rep_dim: Dimensionality of the shared representation ``phi``.
        phi_layers: Hidden layers for the representation MLP.
        head_layers: Hidden layers for the outcome and effect heads.
        disc_layers: Hidden layers for the discriminator.
        activation: Activation function used in all networks.
        phi_dropout: Dropout probability for the representation MLP.
        head_dropout: Dropout probability for the outcome and effect heads.
        disc_dropout: Dropout probability for the discriminator.
        residual: Enable residual connections in all MLPs.
        device: Device string, defaults to CUDA if available.
        seed: Optional random seed for reproducibility.
        epochs: Number of training epochs.
        alpha_out: Weight of the outcome loss.
        beta_cons: Weight of the consistency term.
        gamma_adv: Weight of the adversarial loss.
        lr_g: Learning rate for generator parameters.
        lr_d: Learning rate for the discriminator.
        optimizer: Optimiser used for both generator and discriminator. Either
            a string (``"adam"``, ``"sgd"``, ``"adamw"`` or ``"rmsprop"``) or an
            optimiser class.
        opt_g_kwargs: Optional dictionary with extra arguments for the generator
            optimiser.
        opt_d_kwargs: Optional dictionary with extra arguments for the
            discriminator optimiser.
        lr_scheduler: Optional learning rate scheduler used for both optimisers.
            May be a string (``"step"``, ``"multistep"``, ``"exponential``",
            ``"cosine"`` or ``"plateau"``) or a scheduler class.
        sched_g_kwargs: Extra keyword arguments for the generator scheduler.
        sched_d_kwargs: Extra keyword arguments for the discriminator scheduler.
        grad_clip: Maximum gradient norm.
        warm_start: Number of epochs to train without adversary.
        use_wgan_gp: Use WGAN-GP loss for the discriminator.
        spectral_norm: Apply spectral normalization to all linear layers.
        feature_matching: Add feature matching loss.
        label_smoothing: Use label smoothing for the adversary.
        instance_noise: Inject Gaussian noise into real and fake samples.
        gradient_reversal: Use gradient reversal instead of the adversary.
        ttur: Enable the Two Time-Update Rule which freezes the discriminator
            once its loss drops below a threshold, allowing the generator to
            catch up.
        lambda_gp: Coefficient of the gradient penalty term used in the
            WGAN-GP objective. Only effective when ``use_wgan_gp`` is ``True``.
        eta_fm: Weight of the feature matching term.
        grl_weight: Scale of the gradient reversal layer.
        tensorboard_logdir: Directory for TensorBoard logs.
        weight_clip: Optional weight clipping for the discriminator.
        val_data: Tuple ``(X, mu0, mu1)`` for validation PEHE.
        risk_data: Optional tuple ``(X, T, Y)`` to early-stop on orthogonal risk
            when counterfactuals are unavailable.
        risk_folds: Number of cross-fitting folds for ``risk_data``.
        nuisance_propensity_epochs: Training epochs for the propensity model.
        nuisance_outcome_epochs: Training epochs for the outcome models.
        nuisance_early_stop: Early-stopping patience for nuisance models.
        patience: Early-stopping patience based on validation metric.
        verbose: Print progress every 5 epochs.
        return_history: If ``True`` also return training history.

    Returns:
        The trained ``ACX`` model. If ``return_history`` is ``True`` the
        function instead returns ``(model, history)`` where ``history`` is a
        list of :class:`EpochStats`.
    """
    if model_config is not None:
        if model_config.p != p:
            raise ValueError("p does not match model_config.p")
        rep_dim = model_config.rep_dim
        phi_layers = model_config.phi_layers
        head_layers = model_config.head_layers
        disc_layers = model_config.disc_layers
        activation = model_config.activation
        phi_dropout = model_config.phi_dropout
        head_dropout = model_config.head_dropout
        disc_dropout = model_config.disc_dropout
        residual = model_config.residual

    if training_config is not None:
        epochs = training_config.epochs
        alpha_out = training_config.alpha_out
        beta_cons = training_config.beta_cons
        gamma_adv = training_config.gamma_adv
        lr_g = training_config.lr_g
        lr_d = training_config.lr_d
        optimizer = training_config.optimizer
        opt_g_kwargs = training_config.opt_g_kwargs
        opt_d_kwargs = training_config.opt_d_kwargs
        lr_scheduler = training_config.lr_scheduler
        sched_g_kwargs = training_config.sched_g_kwargs
        sched_d_kwargs = training_config.sched_d_kwargs
        grad_clip = training_config.grad_clip
        warm_start = training_config.warm_start
        use_wgan_gp = training_config.use_wgan_gp
        spectral_norm = training_config.spectral_norm
        feature_matching = training_config.feature_matching
        label_smoothing = training_config.label_smoothing
        instance_noise = training_config.instance_noise
        gradient_reversal = training_config.gradient_reversal
        ttur = training_config.ttur
        lambda_gp = training_config.lambda_gp
        eta_fm = training_config.eta_fm
        grl_weight = training_config.grl_weight
        tensorboard_logdir = training_config.tensorboard_logdir
        weight_clip = training_config.weight_clip
        val_data = training_config.val_data
        risk_data = training_config.risk_data
        risk_folds = training_config.risk_folds
        nuisance_propensity_epochs = training_config.nuisance_propensity_epochs
        nuisance_outcome_epochs = training_config.nuisance_outcome_epochs
        nuisance_early_stop = training_config.nuisance_early_stop
        patience = training_config.patience
        verbose = training_config.verbose
        return_history = training_config.return_history

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        set_seed(seed)

    if grad_clip is not None and grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")
    if weight_clip is not None and weight_clip <= 0:
        raise ValueError("weight_clip must be positive")

    # sanity check for feature dimension mismatches
    feat_dim = None
    try:
        feat_dim = loader.dataset[0][0].shape[-1]
    except Exception:
        # ignore datasets without random access
        pass
    if feat_dim is not None and feat_dim != p:
        raise ValueError(
            f"Input dimension mismatch: dataset has {feat_dim} features but p={p}"
        )

    activation_fn = _get_activation(activation)

    model = ACX(
        p,
        rep_dim=rep_dim,
        phi_layers=phi_layers,
        head_layers=head_layers,
        disc_layers=disc_layers,
        activation=activation_fn,
        phi_dropout=phi_dropout,
        head_dropout=head_dropout,
        disc_dropout=disc_dropout,
        residual=residual,
    ).to(device)
    if spectral_norm:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.utils.spectral_norm(m)

    if isinstance(optimizer, str):
        opt_name = optimizer.lower()
        optimisers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if opt_name not in optimisers:
            raise ValueError(f"Unknown optimizer '{optimizer}'")
        opt_cls = optimisers[opt_name]
    else:
        opt_cls = optimizer
    opt_g_kwargs = opt_g_kwargs or {}
    opt_d_kwargs = opt_d_kwargs or {}

    opt_g = opt_cls(
        list(model.phi.parameters())
        + list(model.mu0.parameters())
        + list(model.mu1.parameters())
        + list(model.tau.parameters()),
        lr=lr_g,
        **opt_g_kwargs,
    )
    opt_d = opt_cls(model.disc.parameters(), lr=lr_d, **opt_d_kwargs)

    sched_g_kwargs = sched_g_kwargs or {}
    sched_d_kwargs = sched_d_kwargs or {}
    sched_g = sched_d = None
    if lr_scheduler is not None:
        if isinstance(lr_scheduler, str):
            name = lr_scheduler.lower()
            schedulers = {
                "step": (
                    torch.optim.lr_scheduler.StepLR,
                    {"step_size": 1, "gamma": 0.9},
                ),
                "multistep": (
                    torch.optim.lr_scheduler.MultiStepLR,
                    {"milestones": [10, 20], "gamma": 0.1},
                ),
                "exponential": (torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.9}),
                "cosine": (torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": 10}),
                "plateau": (
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                    {"mode": "min", "factor": 0.1, "patience": 10},
                ),
            }
            if name not in schedulers:
                raise ValueError(f"Unknown lr scheduler '{lr_scheduler}'")
            sched_cls, default_args = schedulers[name]
            # Merge default arguments with user-supplied arguments
            sched_g_kwargs = {**default_args, **sched_g_kwargs}
            sched_d_kwargs = {**default_args, **sched_d_kwargs}
        else:
            sched_cls = lr_scheduler
        sched_g = sched_cls(opt_g, **sched_g_kwargs)
        sched_d = sched_cls(opt_d, **sched_d_kwargs)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    history: History = []
    writer = SummaryWriter(tensorboard_logdir) if tensorboard_logdir else None
    best_val = float("inf")
    epochs_no_improve = 0
    best_state = None
    freeze_d = False

    if risk_data is not None:
        Xr, Tr, Yr = (v.to(device) for v in risk_data)
        e_hat_val, mu0_val, mu1_val = _estimate_nuisances(
            Xr,
            Tr,
            Yr,
            folds=risk_folds,
            device=device,
            propensity_epochs=nuisance_propensity_epochs,
            outcome_epochs=nuisance_outcome_epochs,
            early_stop=nuisance_early_stop,
        )

    for epoch in range(epochs):
        model.train()
        loss_d_sum = loss_g_sum = 0.0
        loss_y_sum = loss_cons_sum = loss_adv_sum = 0.0
        batch_count = 0
        for Xb, Tb, Yb in loader:
            Xb, Tb, Yb = Xb.to(device), Tb.to(device), Yb.to(device)
            if Tb.ndim == 1:
                Tb = Tb.unsqueeze(-1)
            if Yb.ndim == 1:
                Yb = Yb.unsqueeze(-1)
            Tb = Tb.float()
            Yb = Yb.float()
            with torch.no_grad():
                hb_det, m0_det, m1_det, _ = model(Xb)

            # warm start: train generator without adversary
            if warm_start > 0 and epoch < warm_start:
                hb, m0, m1, _ = model(Xb)
                loss_y = mse(torch.where(Tb.bool(), m1, m0), Yb)
                opt_g.zero_grad()
                loss_y.backward()
                # clip before stepping to avoid exploding gradients
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt_g.step()
                loss_y_sum += loss_y.item()
                loss_g_sum += loss_y.item()
                batch_count += 1
                continue

            # ------------- discriminator update -------------------------
            if not gradient_reversal:
                with torch.no_grad():
                    Ycf = torch.where(Tb.bool(), m0_det, m1_det)
                    Yb_disc = Yb
                    if instance_noise:
                        noise = torch.randn_like(Ycf) * max(
                            0.0, 0.2 * (1 - epoch / epochs)
                        )
                        Ycf = Ycf + noise
                        Yb_disc = Yb_disc + noise
                real_logits = model.discriminator(hb_det, Yb_disc, Tb)
                fake_logits = model.discriminator(hb_det, Ycf, Tb)
                if use_wgan_gp:
                    wdist = fake_logits.mean() - real_logits.mean()
                    gp = 0.0
                    if lambda_gp > 0:
                        eps = torch.rand_like(Yb_disc)
                        interpolates = eps * Yb_disc + (1 - eps) * Ycf
                        interpolates.requires_grad_(True)
                        interp_logits = model.discriminator(hb_det, interpolates, Tb)
                        grads = torch.autograd.grad(
                            outputs=interp_logits,
                            inputs=interpolates,
                            grad_outputs=torch.ones_like(interp_logits),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
                    loss_d = wdist + gp
                else:
                    real_lbl = torch.ones_like(real_logits)
                    fake_lbl = torch.zeros_like(fake_logits)
                    if label_smoothing:
                        real_lbl = real_lbl * 0.9
                        fake_lbl = fake_lbl + 0.1
                    loss_d = bce(real_logits, real_lbl) + bce(fake_logits, fake_lbl)
                if not freeze_d:
                    opt_d.zero_grad()
                    loss_d.backward()
                    opt_d.step()
                    if weight_clip is not None:
                        for p_ in model.disc.parameters():
                            p_.data.clamp_(-weight_clip, weight_clip)
                loss_d_sum += loss_d.item()

            # ------------- generator update -----------------------------
            hb, m0, m1, tau = model(Xb)
            m_obs = torch.where(Tb.bool(), m1, m0)
            loss_y = mse(m_obs, Yb)
            loss_cons = mse(tau, m1 - m0)
            Ycf = torch.where(Tb.bool(), m0, m1)
            loss_adv = torch.tensor(0.0, device=device)
            if not gradient_reversal:
                fake_logits = model.discriminator(hb, Ycf, Tb)
                if use_wgan_gp:
                    loss_adv = -fake_logits.mean()
                else:
                    real_lbl = torch.ones_like(fake_logits)
                    if label_smoothing:
                        real_lbl = real_lbl * 0.9
                    loss_adv = bce(fake_logits, real_lbl)

            loss_g = alpha_out * loss_y + beta_cons * loss_cons + gamma_adv * loss_adv

            if feature_matching:
                with torch.no_grad():
                    real_f = model.disc_features(hb.detach(), Yb, Tb)
                fake_f = model.disc_features(hb, Ycf, Tb)
                loss_fm = ((real_f.mean(0) - fake_f.mean(0)) ** 2).mean()
                loss_g += eta_fm * loss_fm

            if gradient_reversal:
                t_logits = model.discriminator(
                    grad_reverse(hb, grl_weight),
                    Yb,
                    Tb,
                )
                loss_grl = bce(t_logits, Tb)
                loss_g += loss_grl

            opt_g.zero_grad()
            loss_g.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt_g.step()
            loss_g_sum += loss_g.item()
            loss_y_sum += loss_y.item()
            loss_cons_sum += loss_cons.item()
            loss_adv_sum += loss_adv.item()
            batch_count += 1

        stats = EpochStats(
            epoch=epoch,
            loss_d=loss_d_sum / max(1, batch_count),
            loss_g=loss_g_sum / max(1, batch_count),
            loss_y=loss_y_sum / max(1, batch_count),
            loss_cons=loss_cons_sum / max(1, batch_count),
            loss_adv=loss_adv_sum / max(1, batch_count),
        )
        if val_data is not None:
            Xv, mu0v, mu1v = (v.to(device) for v in val_data)
            if mu0v.ndim == 1:
                mu0v = mu0v.unsqueeze(-1)
            if mu1v.ndim == 1:
                mu1v = mu1v.unsqueeze(-1)
            val_pehe = evaluate(model, Xv, mu0v, mu1v)
            stats.val_pehe = val_pehe
            if writer:
                writer.add_scalar("val_pehe", val_pehe, epoch)
            metric = val_pehe
        elif risk_data is not None:
            Xr, Tr, Yr = (v.to(device) for v in risk_data)
            with torch.no_grad():
                _, _, _, tau_r = model(Xr)
            metric = _orthogonal_risk(tau_r, Yr, Tr, e_hat_val, mu0_val, mu1_val)
            stats.val_pehe = metric
            if writer:
                writer.add_scalar("val_risk", metric, epoch)
        else:
            metric = stats.loss_g

        if metric < best_val:
            best_val = metric
            epochs_no_improve = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            epochs_no_improve += 1

        history.append(stats)
        if writer:
            writer.add_scalar("loss/discriminator", stats.loss_d, epoch)
            writer.add_scalar("loss/generator", stats.loss_g, epoch)

        if sched_g:
            if isinstance(sched_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_s = (
                    stats.val_pehe
                    if (val_data is not None or risk_data is not None)
                    else stats.loss_g
                )
                sched_g.step(metric_s)
            else:
                sched_g.step()
        if sched_d:
            if isinstance(sched_d, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_s = (
                    stats.val_pehe
                    if (val_data is not None or risk_data is not None)
                    else stats.loss_d
                )
                sched_d.step(metric_s)
            else:
                sched_d.step()

        if ttur:
            freeze_d = stats.loss_d < 0.3

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            msg = (
                f"epoch {epoch:2d} Ly={stats.loss_y:.3f} "
                f"Lcons={stats.loss_cons:.3f} Ladv={stats.loss_adv:.3f}"
            )
            if val_data is not None:
                msg += f" val_pehe={stats.val_pehe:.3f}"
            elif risk_data is not None:
                msg += f" val_risk={stats.val_pehe:.3f}"
            print(msg)

        if patience > 0 and epochs_no_improve >= patience:
            break

    if writer:
        writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return (model, history) if return_history else model
