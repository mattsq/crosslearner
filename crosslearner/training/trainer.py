from __future__ import annotations

from typing import Optional, Tuple
from collections import OrderedDict
from torch.nn.utils import stateless

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import ModelConfig, TrainingConfig
from .history import EpochStats, History
from ..models.acx import ACX, _get_activation
from ..training.nuisance import estimate_nuisances
from ..evaluation.evaluate import evaluate
from ..utils import set_seed, default_device, apply_spectral_norm
from .grl import grad_reverse


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Return the RBF Maximum Mean Discrepancy between two samples."""

    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    dist_xx = torch.cdist(x, x).pow(2)
    dist_yy = torch.cdist(y, y).pow(2)
    dist_xy = torch.cdist(x, y).pow(2)
    k_xx = torch.exp(-dist_xx / (2 * sigma**2))
    k_yy = torch.exp(-dist_yy / (2 * sigma**2))
    k_xy = torch.exp(-dist_xy / (2 * sigma**2))
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


class ACXTrainer:
    """Trainer class encapsulating the ACX training loop."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        *,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device or default_device()
        if seed is not None:
            set_seed(seed)
        act_fn = _get_activation(model_cfg.activation)
        self.model = ACX(
            model_cfg.p,
            rep_dim=model_cfg.rep_dim,
            disentangle=model_cfg.disentangle,
            rep_dim_c=model_cfg.rep_dim_c,
            rep_dim_a=model_cfg.rep_dim_a,
            rep_dim_i=model_cfg.rep_dim_i,
            phi_layers=model_cfg.phi_layers,
            head_layers=model_cfg.head_layers,
            disc_layers=model_cfg.disc_layers,
            activation=act_fn,
            phi_dropout=model_cfg.phi_dropout,
            head_dropout=model_cfg.head_dropout,
            disc_dropout=model_cfg.disc_dropout,
            residual=model_cfg.residual,
            phi_residual=model_cfg.phi_residual,
            head_residual=model_cfg.head_residual,
            disc_residual=model_cfg.disc_residual,
            disc_pack=model_cfg.disc_pack,
            batch_norm=model_cfg.batch_norm,
        ).to(self.device)
        if train_cfg.spectral_norm:
            apply_spectral_norm(self.model)

        self.ema_model: ACX | None = None
        if train_cfg.ema_decay is not None:
            self.ema_model = ACX(
                model_cfg.p,
                rep_dim=model_cfg.rep_dim,
                disentangle=model_cfg.disentangle,
                rep_dim_c=model_cfg.rep_dim_c,
                rep_dim_a=model_cfg.rep_dim_a,
                rep_dim_i=model_cfg.rep_dim_i,
                phi_layers=model_cfg.phi_layers,
                head_layers=model_cfg.head_layers,
                disc_layers=model_cfg.disc_layers,
                activation=act_fn,
                phi_dropout=model_cfg.phi_dropout,
                head_dropout=model_cfg.head_dropout,
                disc_dropout=model_cfg.disc_dropout,
                residual=model_cfg.residual,
                phi_residual=model_cfg.phi_residual,
                head_residual=model_cfg.head_residual,
                disc_residual=model_cfg.disc_residual,
                disc_pack=model_cfg.disc_pack,
                batch_norm=model_cfg.batch_norm,
            ).to(self.device)
            if train_cfg.spectral_norm:
                apply_spectral_norm(self.ema_model)
            self.ema_model.load_state_dict(self.model.state_dict())
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self._ema_params = dict(self.ema_model.named_parameters())

        self._rep_means: dict[int, torch.Tensor] | None = None
        self._rep_vars: dict[int, torch.Tensor] | None = None

    def _update_ema(self) -> None:
        decay = self.train_cfg.ema_decay
        if self.ema_model is None or decay is None:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name.startswith("disc."):
                    continue
                ema_param = self._ema_params[name]
                ema_param.mul_(decay).add_(param.detach(), alpha=1 - decay)

    def _adjust_regularization(self, loss_d: float) -> None:
        cfg = self.train_cfg
        if not cfg.adaptive_reg:
            return
        adv = (cfg.adv_loss or "bce").lower()
        if adv != "bce":
            return
        if loss_d < cfg.d_reg_lower:
            cfg.lambda_gp = min(cfg.lambda_gp * cfg.reg_factor, cfg.lambda_gp_max)
        elif loss_d > cfg.d_reg_upper:
            cfg.lambda_gp = max(cfg.lambda_gp / cfg.reg_factor, cfg.lambda_gp_min)

    def _pack_inputs(
        self, h: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pack = max(1, int(self.model_cfg.disc_pack))
        if pack <= 1:
            return h, y, t
        b = (h.size(0) // pack) * pack
        h = h[:b].reshape(b // pack, -1)
        y = y[:b].reshape(b // pack, -1)
        t = t[:b].reshape(b // pack, -1)
        return h, y, t

    def _sample_negatives(self, t: torch.Tensor) -> torch.Tensor:
        """Return indices of negative samples from the opposite treatment group."""
        t = t.view(-1)
        n = t.size(0)
        idx0 = torch.where(t == 0)[0]
        idx1 = torch.where(t == 1)[0]

        if idx0.numel() == 0 or idx1.numel() == 0:
            return torch.randint(n, (n,), device=t.device)

        neg_idx = torch.empty(n, dtype=torch.long, device=t.device)
        num0 = idx0.numel()
        num1 = idx1.numel()
        mask0 = t == 0
        mask1 = t == 1
        if mask0.any():
            neg_idx[mask0] = idx1[torch.randint(num1, (mask0.sum(),), device=t.device)]
        if mask1.any():
            neg_idx[mask1] = idx0[torch.randint(num0, (mask1.sum(),), device=t.device)]
        return neg_idx

    def _unrolled_logits(
        self,
        hb: torch.Tensor,
        yb: torch.Tensor,
        ycf: torch.Tensor,
        tb: torch.Tensor,
        bce: nn.Module,
        use_wgan: bool,
        use_hinge: bool,
        use_ls: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = self.train_cfg.unrolled_steps
        if steps <= 0:
            hb_p, ycf_p, tb_p = self._pack_inputs(hb, ycf, tb)
            logits = self.model.discriminator(hb_p, ycf_p, tb_p)
            feats = self.model.disc_features(hb_p, ycf_p, tb_p)
            return logits, feats
        disc = self.model.disc
        params = OrderedDict(
            (n, p.detach().clone().requires_grad_(True))
            for n, p in disc.named_parameters()
        )
        for _ in range(steps):
            hb_r, y_r, t_r = self._pack_inputs(hb.detach(), yb.detach(), tb)
            hb_f, y_f, t_f = self._pack_inputs(hb.detach(), ycf.detach(), tb)
            real_logits = stateless.functional_call(
                disc, params, (torch.cat([hb_r, y_r, t_r], dim=1),)
            )
            fake_logits = stateless.functional_call(
                disc, params, (torch.cat([hb_f, y_f, t_f], dim=1),)
            )
            if use_wgan:
                loss_step = fake_logits.mean() - real_logits.mean()
            elif use_hinge:
                loss_step = (
                    torch.relu(1 - real_logits).mean()
                    + torch.relu(1 + fake_logits).mean()
                )
            elif use_ls:
                mse_adv = nn.MSELoss()
                loss_step = mse_adv(
                    real_logits, torch.ones_like(real_logits)
                ) + mse_adv(fake_logits, torch.zeros_like(fake_logits))
            else:
                real_lbl = torch.ones_like(real_logits)
                fake_lbl = torch.zeros_like(fake_logits)
                if self.train_cfg.label_smoothing:
                    real_lbl = real_lbl * 0.9
                    fake_lbl = fake_lbl + 0.1
                loss_step = bce(real_logits, real_lbl) + bce(fake_logits, fake_lbl)
            grads = torch.autograd.grad(
                loss_step, tuple(params.values()), create_graph=True
            )
            params = OrderedDict(
                (
                    n,
                    (p - self.train_cfg.lr_d * g).detach().requires_grad_(True),
                )
                for (n, p), g in zip(params.items(), grads)
            )
        hb_p, ycf_p, tb_p = self._pack_inputs(hb, ycf, tb)
        inp = torch.cat([hb_p, ycf_p, tb_p], dim=1)
        logits = stateless.functional_call(disc, params, (inp,))
        net_params = OrderedDict(
            (k.split("blocks.", 1)[1], v)
            for k, v in params.items()
            if k.startswith("blocks.")
        )
        feats = stateless.functional_call(disc.net[:-1], net_params, (inp,))
        return logits, feats

    def _make_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        cfg = self.train_cfg
        model = self.model

        if isinstance(cfg.optimizer, str):
            opt_name = cfg.optimizer.lower()
            optimisers = {
                "adam": torch.optim.Adam,
                "adamw": torch.optim.AdamW,
                "sgd": torch.optim.SGD,
                "rmsprop": torch.optim.RMSprop,
            }
            if opt_name not in optimisers:
                raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")
            opt_cls = optimisers[opt_name]
        else:
            opt_cls = cfg.optimizer

        opt_g_kwargs = cfg.opt_g_kwargs or {}
        opt_d_kwargs = cfg.opt_d_kwargs or {}

        opt_g = opt_cls(
            list(model.phi.parameters())
            + list(model.mu0.parameters())
            + list(model.mu1.parameters())
            + list(model.tau.parameters())
            + list(model.prop.parameters())
            + [model.epsilon],
            lr=cfg.lr_g,
            **opt_g_kwargs,
        )
        disc_params = list(model.disc.parameters())
        if model.disentangle:
            disc_params += list(model.adv_t.parameters()) + list(
                model.adv_y.parameters()
            )
        opt_d = opt_cls(disc_params, lr=cfg.lr_d, **opt_d_kwargs)
        return opt_g, opt_d

    def _make_schedulers(
        self,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
    ) -> Tuple[
        Optional[torch.optim.lr_scheduler._LRScheduler],
        Optional[torch.optim.lr_scheduler._LRScheduler],
    ]:
        cfg = self.train_cfg
        lr_scheduler = cfg.lr_scheduler
        sched_g_kwargs = cfg.sched_g_kwargs or {}
        sched_d_kwargs = cfg.sched_d_kwargs or {}
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
                    "exponential": (
                        torch.optim.lr_scheduler.ExponentialLR,
                        {"gamma": 0.9},
                    ),
                    "cosine": (
                        torch.optim.lr_scheduler.CosineAnnealingLR,
                        {"T_max": 10},
                    ),
                    "plateau": (
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                        {"mode": "min", "factor": 0.1, "patience": 10},
                    ),
                }
                if name not in schedulers:
                    raise ValueError(f"Unknown lr scheduler '{lr_scheduler}'")
                sched_cls, default_args = schedulers[name]
                sched_g_kwargs = {**default_args, **sched_g_kwargs}
                sched_d_kwargs = {**default_args, **sched_d_kwargs}
            else:
                sched_cls = lr_scheduler
            sched_g = sched_cls(opt_g, **sched_g_kwargs)
            sched_d = sched_cls(opt_d, **sched_d_kwargs)
        return sched_g, sched_d

    def _validate_inputs(self, loader: DataLoader) -> None:
        cfg = self.train_cfg
        if cfg.grad_clip is not None and cfg.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")
        if cfg.weight_clip is not None and cfg.weight_clip <= 0:
            raise ValueError("weight_clip must be positive")

        feat_dim = None
        try:
            feat_dim = loader.dataset[0][0].shape[-1]
        except Exception:
            pass
        if feat_dim is not None and feat_dim != self.model_cfg.p:
            raise ValueError(
                f"Input dimension mismatch: dataset has {feat_dim} features but p={self.model_cfg.p}"
            )

    def _estimate_risk(
        self,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        cfg = self.train_cfg
        device = self.device
        if cfg.risk_data is None:
            return None
        Xr, Tr, Yr = (v.to(device) for v in cfg.risk_data)
        return estimate_nuisances(
            Xr,
            Tr,
            Yr,
            folds=cfg.risk_folds,
            device=device,
            propensity_epochs=cfg.nuisance_propensity_epochs,
            outcome_epochs=cfg.nuisance_outcome_epochs,
            early_stop=cfg.nuisance_early_stop,
        )

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
        bce: nn.Module,
        mse: nn.Module,
        *,
        freeze_d: bool,
    ) -> EpochStats:
        cfg = self.train_cfg
        model = self.model
        device = self.device
        adv = (cfg.adv_loss or "bce").lower()
        use_wgan = cfg.use_wgan_gp or adv == "wgan-gp"
        use_hinge = adv == "hinge"
        use_ls = adv == "lsgan"

        loss_d_sum = loss_g_sum = 0.0
        loss_y_sum = loss_cons_sum = loss_adv_sum = 0.0
        batch_count = 0
        rep_sums = {
            0: torch.zeros(self.model.rep_dim, device=device),
            1: torch.zeros(self.model.rep_dim, device=device),
        }
        rep_sq_sums = {
            0: torch.zeros(self.model.rep_dim, device=device),
            1: torch.zeros(self.model.rep_dim, device=device),
        }
        rep_counts = {0: 0, 1: 0}

        for Xb, Tb, Yb in loader:
            Xb, Tb, Yb = Xb.to(device), Tb.to(device), Yb.to(device)
            if Tb.ndim == 1:
                Tb = Tb.unsqueeze(-1)
            if Yb.ndim == 1:
                Yb = Yb.unsqueeze(-1)
            Tb = Tb.float()
            Yb = Yb.float()
            hb, m0, m1, tau = model(Xb)
            hb_det = hb.detach()
            m0_det = m0.detach()
            m1_det = m1.detach()
            rep_pen = torch.tensor(0.0, device=device)
            if cfg.rep_consistency_weight > 0:
                t_mask = Tb.view(-1) > 0.5
                for g, mask in ((1, t_mask), (0, ~t_mask)):
                    if mask.any():
                        h_g = hb_det[mask]
                        rep_sums[g] += h_g.sum(0)
                        rep_sq_sums[g] += (h_g * h_g).sum(0)
                        rep_counts[g] += h_g.size(0)
                        if (
                            self._rep_means is not None
                            and self._rep_means.get(g) is not None
                        ):
                            mean_g = h_g.mean(0)
                            var_g = h_g.var(0, unbiased=False)
                            rep_pen = (
                                rep_pen
                                + F.mse_loss(mean_g, self._rep_means[g])
                                + F.mse_loss(var_g, self._rep_vars[g])
                            )

            if cfg.warm_start > 0 and epoch < cfg.warm_start:
                loss_y = mse(torch.where(Tb.bool(), m1, m0), Yb)
                opt_g.zero_grad(set_to_none=True)
                loss_y.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt_g.step()
                self._update_ema()
                loss_y_sum += loss_y.item()
                loss_g_sum += loss_y.item()
                batch_count += 1
                continue

            if not cfg.gradient_reversal:
                with torch.no_grad():
                    Ycf_base = torch.where(Tb.bool(), m0_det, m1_det)
                    Yb_disc_base = Yb
                    if cfg.instance_noise:
                        noise = torch.randn_like(Ycf_base) * max(
                            0.0, 0.2 * (1 - epoch / cfg.epochs)
                        )
                        Ycf_base = Ycf_base + noise
                        Yb_disc_base = Yb_disc_base + noise

                for _ in range(max(1, int(cfg.disc_steps))):
                    hb_aug = hb_det
                    if cfg.disc_aug_prob > 0:
                        hb_aug = F.dropout(hb_aug, p=cfg.disc_aug_prob, training=True)
                    if cfg.disc_aug_noise > 0:
                        hb_aug = hb_aug + torch.randn_like(hb_aug) * cfg.disc_aug_noise
                    Ycf = Ycf_base
                    Yb_disc = Yb_disc_base
                    hb_r, y_r, t_r = self._pack_inputs(hb_aug, Yb_disc, Tb)
                    hb_f, y_f, t_f = self._pack_inputs(hb_aug, Ycf, Tb)
                    real_logits = model.discriminator(hb_r, y_r, t_r)
                    fake_logits = model.discriminator(hb_f, y_f, t_f)
                    lbl_shape = real_logits.shape
                    ones_real = torch.ones(lbl_shape, device=device)
                    zeros_fake = torch.zeros(lbl_shape, device=device)

                    if use_wgan:
                        wdist = fake_logits.mean() - real_logits.mean()
                        gp = 0.0
                        if cfg.lambda_gp > 0:
                            eps = torch.rand_like(Yb_disc)
                            interpolates = eps * Yb_disc + (1 - eps) * Ycf
                            interpolates.requires_grad_(True)
                            h_i, y_i, t_i = self._pack_inputs(hb_aug, interpolates, Tb)
                            interp_logits = model.discriminator(h_i, y_i, t_i)
                            grads = torch.autograd.grad(
                                outputs=interp_logits,
                                inputs=interpolates,
                                grad_outputs=ones_real,
                                create_graph=True,
                                only_inputs=True,
                            )[0]
                            gp = (
                                (grads.norm(2, dim=1) - 1) ** 2
                            ).mean() * cfg.lambda_gp
                        loss_d = wdist + gp
                    elif use_hinge:
                        loss_d = (
                            torch.relu(1 - real_logits).mean()
                            + torch.relu(1 + fake_logits).mean()
                        )
                    elif use_ls:
                        mse_adv = nn.MSELoss()
                        loss_d = mse_adv(real_logits, ones_real) + mse_adv(
                            fake_logits, zeros_fake
                        )
                    else:
                        real_lbl = ones_real
                        fake_lbl = zeros_fake
                        if cfg.label_smoothing:
                            real_lbl = real_lbl * 0.9
                            fake_lbl = fake_lbl + 0.1
                        loss_d = bce(real_logits, real_lbl) + bce(fake_logits, fake_lbl)

                    if model.disentangle and (
                        cfg.adv_t_weight > 0 or cfg.adv_y_weight > 0
                    ):
                        zc_d, za_d, zi_d = model.split(hb_aug)
                        if cfg.adv_t_weight > 0:
                            t_pred = model.adv_t_pred(zc_d.detach(), za_d.detach())
                            loss_d = loss_d + cfg.adv_t_weight * bce(t_pred, Tb)
                        if cfg.adv_y_weight > 0:
                            y_pred = model.adv_y_pred(zc_d.detach(), zi_d.detach())
                            loss_d = loss_d + cfg.adv_y_weight * mse(y_pred, Yb)

                    if cfg.r1_gamma > 0:
                        hb_r1 = hb_aug.detach().requires_grad_(True)
                        y_r1 = Yb_disc.detach().requires_grad_(True)
                        h1, y1, t1 = self._pack_inputs(hb_r1, y_r1, Tb)
                        r1_logits = model.discriminator(h1, y1, t1)
                        grads = torch.autograd.grad(
                            r1_logits.sum(), [hb_r1, y_r1], create_graph=True
                        )
                        penalty = (
                            torch.cat([g.view(g.size(0), -1) for g in grads], dim=1)
                            .pow(2)
                            .sum(1)
                            .mean()
                        )
                        loss_d = loss_d + 0.5 * cfg.r1_gamma * penalty

                    if cfg.r2_gamma > 0:
                        hb_r2 = hb_aug.detach().requires_grad_(True)
                        y_r2 = Ycf.detach().requires_grad_(True)
                        h2, y2, t2 = self._pack_inputs(hb_r2, y_r2, Tb)
                        r2_logits = model.discriminator(h2, y2, t2)
                        grads = torch.autograd.grad(
                            r2_logits.sum(), [hb_r2, y_r2], create_graph=True
                        )
                        penalty = (
                            torch.cat([g.view(g.size(0), -1) for g in grads], dim=1)
                            .pow(2)
                            .sum(1)
                            .mean()
                        )
                        loss_d = loss_d + 0.5 * cfg.r2_gamma * penalty

                    if not freeze_d:
                        opt_d.zero_grad(set_to_none=True)
                        loss_d.backward()
                        opt_d.step()
                        if cfg.weight_clip is not None:
                            for p_ in model.disc.parameters():
                                p_.data.clamp_(-cfg.weight_clip, cfg.weight_clip)
                    loss_d_sum += loss_d.item()

            # Reuse outputs from earlier forward pass
            prop = model.propensity(hb)
            m_obs = torch.where(Tb.bool(), m1, m0)
            loss_y = mse(m_obs, Yb)
            loss_cons = mse(tau, m1 - m0)
            loss_prop = bce(prop, Tb)
            eps = model.epsilon
            q_tilde = m_obs + eps * (Tb - prop) / (prop * (1 - prop) + 1e-8)
            loss_dr = mse(Yb, q_tilde)
            loss_noise = torch.tensor(0.0, device=device)
            if cfg.noise_consistency_weight > 0 and cfg.noise_std > 0:
                Xn = Xb + torch.randn_like(Xb) * cfg.noise_std
                _, m0_n, m1_n, tau_n = model(Xn)
                loss_noise = mse(m0_n, m0) + mse(m1_n, m1) + mse(tau_n, tau)
            Ycf = torch.where(Tb.bool(), m0, m1)
            loss_adv = torch.tensor(0.0, device=device)
            loss_contrast = torch.tensor(0.0, device=device)
            if not cfg.gradient_reversal:
                fake_logits, fake_feats = self._unrolled_logits(
                    hb,
                    Yb,
                    Ycf,
                    Tb,
                    bce,
                    use_wgan,
                    use_hinge,
                    use_ls,
                )
                if use_wgan or use_hinge:
                    loss_adv = -fake_logits.mean()
                elif use_ls:
                    mse_adv = nn.MSELoss()
                    loss_adv = mse_adv(fake_logits, torch.ones_like(fake_logits))
                else:
                    real_lbl = torch.ones_like(fake_logits)
                    if cfg.label_smoothing:
                        real_lbl = real_lbl * 0.9
                    loss_adv = bce(fake_logits, real_lbl)

            loss_adv_t = torch.tensor(0.0, device=device)
            loss_adv_y = torch.tensor(0.0, device=device)
            if model.disentangle and (cfg.adv_t_weight > 0 or cfg.adv_y_weight > 0):
                zc, za, zi = model.split(hb)
                if cfg.adv_t_weight > 0:
                    t_pred = model.adv_t_pred(zc.detach(), grad_reverse(za))
                    loss_adv_t = cfg.adv_t_weight * bce(t_pred, Tb)
                if cfg.adv_y_weight > 0:
                    y_pred = model.adv_y_pred(zc.detach(), grad_reverse(zi))
                    loss_adv_y = cfg.adv_y_weight * mse(y_pred, Yb)

            loss_mmd = torch.tensor(0.0, device=device)
            if cfg.mmd_weight > 0:
                h_t = hb[Tb.view(-1) > 0.5]
                h_c = hb[Tb.view(-1) <= 0.5]
                loss_mmd = _mmd_rbf(h_t, h_c, sigma=cfg.mmd_sigma)

            if cfg.contrastive_weight > 0:
                noise = (
                    torch.randn_like(Xb) * cfg.contrastive_noise
                    if cfg.contrastive_noise > 0
                    else 0.0
                )
                h_pos = model.phi(Xb + noise)
                neg_idx = self._sample_negatives(Tb.view(-1))
                h_neg = hb[neg_idx].detach()
                loss_contrast = F.triplet_margin_loss(
                    hb, h_pos, h_neg, margin=cfg.contrastive_margin
                )

            loss_g = (
                cfg.alpha_out * loss_y
                + cfg.beta_cons * loss_cons
                + cfg.gamma_adv * loss_adv
                + loss_adv_t
                + loss_adv_y
                + cfg.contrastive_weight * loss_contrast
                + cfg.mmd_weight * loss_mmd
                + cfg.delta_prop * loss_prop
                + cfg.lambda_dr * loss_dr
                + cfg.noise_consistency_weight * loss_noise
                + cfg.rep_consistency_weight * rep_pen
            )

            if cfg.feature_matching:
                with torch.no_grad():
                    r_h, r_y, r_t = self._pack_inputs(hb.detach(), Yb, Tb)
                    real_f = model.disc_features(r_h, r_y, r_t)
                if not cfg.gradient_reversal:
                    fake_f = fake_feats
                else:
                    f_h, f_y, f_t = self._pack_inputs(hb, Ycf, Tb)
                    fake_f = model.disc_features(f_h, f_y, f_t)
                loss_fm = ((real_f.mean(0) - fake_f.mean(0)) ** 2).mean()
                loss_g += cfg.eta_fm * loss_fm

            if cfg.gradient_reversal:
                gr_h = grad_reverse(hb, cfg.grl_weight)
                h_p, y_p, t_p = self._pack_inputs(gr_h, Yb, Tb)
                t_logits = model.discriminator(h_p, y_p, t_p)
                loss_grl = bce(t_logits, Tb)
                loss_g += loss_grl

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt_g.step()
            self._update_ema()
            loss_g_sum += loss_g.item()
            loss_y_sum += loss_y.item()
            loss_cons_sum += loss_cons.item()
            loss_adv_sum += (loss_adv + loss_adv_t + loss_adv_y).item()
            batch_count += 1

        if cfg.rep_consistency_weight > 0:
            if self._rep_means is None:
                self._rep_means = {0: None, 1: None}
                self._rep_vars = {0: None, 1: None}
            for g in (0, 1):
                if rep_counts[g] > 0:
                    mean_g = rep_sums[g] / rep_counts[g]
                    var_g = rep_sq_sums[g] / rep_counts[g] - mean_g**2
                    if self._rep_means[g] is None:
                        self._rep_means[g] = mean_g.detach()
                        self._rep_vars[g] = var_g.detach()
                    else:
                        mom = cfg.rep_momentum
                        self._rep_means[g].mul_(mom).add_(
                            mean_g.detach(), alpha=1 - mom
                        )
                        self._rep_vars[g].mul_(mom).add_(var_g.detach(), alpha=1 - mom)

        stats = EpochStats(
            epoch=epoch,
            loss_d=loss_d_sum / max(1, batch_count),
            loss_g=loss_g_sum / max(1, batch_count),
            loss_y=loss_y_sum / max(1, batch_count),
            loss_cons=loss_cons_sum / max(1, batch_count),
            loss_adv=loss_adv_sum / max(1, batch_count),
        )
        return stats

    def train(self, loader: DataLoader) -> ACX | Tuple[ACX, History]:
        cfg = self.train_cfg
        model = self.model
        eval_model = self.ema_model if self.ema_model is not None else self.model
        device = self.device

        self._validate_inputs(loader)

        opt_g, opt_d = self._make_optimizers()
        sched_g, sched_d = self._make_schedulers(opt_g, opt_d)

        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        history: History = []
        writer = (
            SummaryWriter(cfg.tensorboard_logdir) if cfg.tensorboard_logdir else None
        )
        best_val = float("inf")
        epochs_no_improve = 0
        best_state = None
        freeze_d = False

        risk_vals = self._estimate_risk()
        if risk_vals is not None:
            e_hat_val, mu0_val, mu1_val = risk_vals

        for epoch in range(cfg.epochs):
            stats = self._train_epoch(
                loader,
                epoch,
                opt_g,
                opt_d,
                bce,
                mse,
                freeze_d=freeze_d,
            )
            if cfg.val_data is not None:
                Xv, mu0v, mu1v = (v.to(device) for v in cfg.val_data)
                if mu0v.ndim == 1:
                    mu0v = mu0v.unsqueeze(-1)
                if mu1v.ndim == 1:
                    mu1v = mu1v.unsqueeze(-1)
                val_pehe = evaluate(eval_model, Xv, mu0v, mu1v)
                stats.val_pehe = val_pehe
                if writer:
                    writer.add_scalar("val_pehe", val_pehe, epoch)
                metric = val_pehe
            elif cfg.risk_data is not None:
                Xr, Tr, Yr = (v.to(device) for v in cfg.risk_data)
                with torch.no_grad():
                    _, _, _, tau_r = eval_model(Xr)
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
                    k: v.detach().cpu().clone()
                    for k, v in eval_model.state_dict().items()
                }
            else:
                epochs_no_improve += 1

            history.append(stats)
            if writer:
                writer.add_scalar("loss/discriminator", stats.loss_d, epoch)
                writer.add_scalar("loss/generator", stats.loss_g, epoch)

            metric_g = (
                stats.val_pehe
                if (cfg.val_data is not None or cfg.risk_data is not None)
                else stats.loss_g
            )
            _scheduler_step(sched_g, metric_g)
            metric_d = (
                stats.val_pehe
                if (cfg.val_data is not None or cfg.risk_data is not None)
                else stats.loss_d
            )
            _scheduler_step(sched_d, metric_d)
            self._adjust_regularization(stats.loss_d)

            if cfg.ttur:
                freeze_d = stats.loss_d < 0.3

            if cfg.verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
                msg = (
                    f"epoch {epoch:2d} Ly={stats.loss_y:.3f} "
                    f"Lcons={stats.loss_cons:.3f} Ladv={stats.loss_adv:.3f}"
                )
                if cfg.val_data is not None:
                    msg += f" val_pehe={stats.val_pehe:.3f}"
                elif cfg.risk_data is not None:
                    msg += f" val_risk={stats.val_pehe:.3f}"
                print(msg)

            if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
                break

        if writer:
            writer.close()
        if best_state is not None:
            model.load_state_dict(best_state)
        elif self.ema_model is not None:
            model.load_state_dict(self.ema_model.state_dict())
        model.eval()
        return (model, history) if cfg.return_history else model


def _scheduler_step(
    scheduler: Optional[
        torch.optim.lr_scheduler._LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    ],
    metric: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


@torch.no_grad()
def _orthogonal_risk(
    tau_hat: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    e_hat: torch.Tensor,
    mu0_hat: torch.Tensor,
    mu1_hat: torch.Tensor,
) -> float:
    mse = nn.MSELoss()
    y_resid = y - torch.where(t.bool(), mu1_hat, mu0_hat)
    loss = mse(y_resid - (t - e_hat) * tau_hat, torch.zeros_like(y))
    return loss.item()
