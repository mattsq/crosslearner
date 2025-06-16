import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Iterable, Callable, Type
from torch.utils.tensorboard import SummaryWriter

from crosslearner.training.history import EpochStats, History
from crosslearner.evaluation.evaluate import evaluate

from crosslearner.models.acx import ACX
from crosslearner.training.grl import grad_reverse


def train_acx(
    loader: DataLoader,
    p: int,
    *,
    rep_dim: int = 64,
    phi_layers: Iterable[int] | None = (128,),
    head_layers: Iterable[int] | None = (64,),
    disc_layers: Iterable[int] | None = (64,),
    activation: str | Callable[[], nn.Module] = "relu",
    device: Optional[str] = None,
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
    patience: int = 0,
    verbose: bool = True,
    return_history: bool = False,
):
    """Train AC-X model with optional GAN tricks.

    Args:
        loader: PyTorch dataloader yielding ``(X, T, Y)`` batches.
        p: Number of covariates.
        rep_dim: Dimensionality of the shared representation ``phi``.
        phi_layers: Hidden layers for the representation MLP.
        head_layers: Hidden layers for the outcome and effect heads.
        disc_layers: Hidden layers for the discriminator.
        activation: Activation function used in all networks.
        device: Device string, defaults to CUDA if available.
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
        gradient_reversal: Replace adversary with gradient reversal.
        ttur: Use Two-Time-Update-Rule with different learning rates.
        lambda_gp: Gradient penalty coefficient for WGAN-GP.
        eta_fm: Weight of the feature matching term.
        grl_weight: Scale of the gradient reversal layer.
        tensorboard_logdir: Directory for TensorBoard logs.
        weight_clip: Optional weight clipping for the discriminator.
        val_data: Tuple ``(X, mu0, mu1)`` for validation PEHE.
        patience: Early-stopping patience based on validation PEHE.
        verbose: Print progress every 5 epochs.
        return_history: If ``True`` also return training history.

    Returns:
        The trained ``ACX`` model or tuple ``(model, history)`` when
        ``return_history`` is ``True``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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

    if isinstance(activation, str):
        act_name = activation.lower()
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "leakyrelu": nn.LeakyReLU,
        }
        if act_name not in activations:
            raise ValueError(f"Unknown activation '{activation}'")
        activation_fn = activations[act_name]
    else:
        activation_fn = activation

    model = ACX(
        p,
        rep_dim=rep_dim,
        phi_layers=phi_layers,
        head_layers=head_layers,
        disc_layers=disc_layers,
        activation=activation_fn,
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
                "step": torch.optim.lr_scheduler.StepLR,
                "multistep": torch.optim.lr_scheduler.MultiStepLR,
                "exponential": torch.optim.lr_scheduler.ExponentialLR,
                "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
                "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            }
            if name not in schedulers:
                raise ValueError(f"Unknown lr scheduler '{lr_scheduler}'")
            sched_cls = schedulers[name]
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
    freeze_d = False

    for epoch in range(epochs):
        loss_d_sum = loss_g_sum = 0.0
        loss_y_sum = loss_cons_sum = loss_adv_sum = 0.0
        batch_count = 0
        for Xb, Tb, Yb in loader:
            Xb, Tb, Yb = Xb.to(device), Tb.to(device), Yb.to(device)
            if Tb.ndim == 1:
                Tb = Tb.unsqueeze(-1)
            if Yb.ndim == 1:
                Yb = Yb.unsqueeze(-1)
            hb, m0, m1, tau = model(Xb)

            if warm_start > 0 and epoch < warm_start:
                loss_y = mse(torch.where(Tb.bool(), m1, m0), Yb)
                opt_g.zero_grad()
                loss_y.backward()
                opt_g.step()
                continue

            # ------------- discriminator update -------------------------
            with torch.no_grad():
                Ycf = torch.where(Tb.bool(), m0, m1).detach()
                Yb_disc = Yb
                if instance_noise:
                    noise = torch.randn_like(Ycf) * max(0.0, 0.2 * (1 - epoch / epochs))
                    Ycf = Ycf + noise
                    Yb_disc = Yb_disc + noise
            real_logits = model.discriminator(hb, Yb_disc, Tb)
            fake_logits = model.discriminator(hb, Ycf, Tb)
            if use_wgan_gp:
                wdist = fake_logits.mean() - real_logits.mean()
                gp = 0.0
                if lambda_gp > 0:
                    eps = torch.rand_like(Yb_disc)
                    interpolates = eps * Yb_disc + (1 - eps) * Ycf
                    interpolates.requires_grad_(True)
                    interp_logits = model.discriminator(hb, interpolates, Tb)
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
                real_f = hb.detach()
                fake_f = hb
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
            if val_pehe < best_val:
                best_val = val_pehe
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        history.append(stats)
        if writer:
            writer.add_scalar("loss/discriminator", stats.loss_d, epoch)
            writer.add_scalar("loss/generator", stats.loss_g, epoch)

        if sched_g:
            if isinstance(sched_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = stats.val_pehe if val_data is not None else stats.loss_g
                sched_g.step(metric)
            else:
                sched_g.step()
        if sched_d:
            if isinstance(sched_d, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = stats.val_pehe if val_data is not None else stats.loss_d
                sched_d.step(metric)
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
            print(msg)

        if patience > 0 and epochs_no_improve >= patience:
            break

    if writer:
        writer.close()
    model.eval()
    return (model, history) if return_history else model
