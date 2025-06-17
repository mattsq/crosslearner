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
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    *,
    device: Optional[str] = None,
    seed: int | None = None,
) -> ACX | tuple[ACX, History]:
    """Train AC-X model with optional GAN tricks.

    Parameters are read from ``model_cfg`` and ``train_cfg``. ``device`` and
    ``seed`` override the defaults from the configuration.

    Args:
        loader: Dataloader yielding ``(X, T, Y)`` batches.
        model_cfg: Architecture options for :class:`ACX`.
        train_cfg: Hyperparameters controlling optimisation.
        device: Optional device string, defaults to CUDA if available.
        seed: Optional random seed for reproducibility.

    Returns:
        The trained ``ACX`` model. If ``return_history`` is ``True`` the
        function instead returns ``(model, history)`` where ``history`` is a
        list of :class:`EpochStats`.
    """
    p = model_cfg.p
    rep_dim = model_cfg.rep_dim
    phi_layers = model_cfg.phi_layers
    head_layers = model_cfg.head_layers
    disc_layers = model_cfg.disc_layers
    activation = model_cfg.activation
    phi_dropout = model_cfg.phi_dropout
    head_dropout = model_cfg.head_dropout
    disc_dropout = model_cfg.disc_dropout
    residual = model_cfg.residual

    epochs = train_cfg.epochs
    alpha_out = train_cfg.alpha_out
    beta_cons = train_cfg.beta_cons
    gamma_adv = train_cfg.gamma_adv
    lr_g = train_cfg.lr_g
    lr_d = train_cfg.lr_d
    optimizer = train_cfg.optimizer
    opt_g_kwargs = train_cfg.opt_g_kwargs
    opt_d_kwargs = train_cfg.opt_d_kwargs
    lr_scheduler = train_cfg.lr_scheduler
    sched_g_kwargs = train_cfg.sched_g_kwargs
    sched_d_kwargs = train_cfg.sched_d_kwargs
    grad_clip = train_cfg.grad_clip
    warm_start = train_cfg.warm_start
    use_wgan_gp = train_cfg.use_wgan_gp
    spectral_norm = train_cfg.spectral_norm
    feature_matching = train_cfg.feature_matching
    label_smoothing = train_cfg.label_smoothing
    instance_noise = train_cfg.instance_noise
    gradient_reversal = train_cfg.gradient_reversal
    ttur = train_cfg.ttur
    lambda_gp = train_cfg.lambda_gp
    eta_fm = train_cfg.eta_fm
    grl_weight = train_cfg.grl_weight
    tensorboard_logdir = train_cfg.tensorboard_logdir
    weight_clip = train_cfg.weight_clip
    val_data = train_cfg.val_data
    risk_data = train_cfg.risk_data
    risk_folds = train_cfg.risk_folds
    nuisance_propensity_epochs = train_cfg.nuisance_propensity_epochs
    nuisance_outcome_epochs = train_cfg.nuisance_outcome_epochs
    nuisance_early_stop = train_cfg.nuisance_early_stop
    patience = train_cfg.patience
    verbose = train_cfg.verbose
    return_history = train_cfg.return_history

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
