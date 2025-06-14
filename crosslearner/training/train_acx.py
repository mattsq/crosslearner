import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from crosslearner.training.history import EpochStats, History

from crosslearner.models.acx import ACX
from crosslearner.training.grl import grad_reverse


def train_acx(
    loader: DataLoader,
    p: int,
    *,
    device: Optional[str] = None,
    epochs: int = 30,
    alpha_out: float = 1.0,
    beta_cons: float = 10.0,
    gamma_adv: float = 1.0,
    lr_g: float = 1e-3,
    lr_d: float = 1e-3,
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
    verbose: bool = True,
    return_history: bool = False,
):
    """Train AC-X model with optional GAN tricks."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = ACX(p).to(device)
    if spectral_norm:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.utils.spectral_norm(m)

    opt_g = torch.optim.Adam(
        list(model.phi.parameters())
        + list(model.mu0.parameters())
        + list(model.mu1.parameters())
        + list(model.tau.parameters()),
        lr=lr_g,
    )
    opt_d = torch.optim.Adam(model.disc.parameters(), lr=lr_d)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    history: History = []

    for epoch in range(epochs):
        loss_d_sum = loss_g_sum = 0.0
        loss_y_sum = loss_cons_sum = loss_adv_sum = 0.0
        batch_count = 0
        for Xb, Tb, Yb in loader:
            Xb, Tb, Yb = Xb.to(device), Tb.to(device), Yb.to(device)
            hb, m0, m1, tau = model(Xb)

            if warm_start > 0 and epoch < warm_start:
                loss_y = mse(torch.where(Tb.bool(), m1, m0), Yb)
                opt_g.zero_grad(); loss_y.backward(); opt_g.step()
                continue

            # ------------- discriminator update -------------------------
            with torch.no_grad():
                Ycf = torch.where(Tb.bool(), m0, m1).detach()
                if instance_noise:
                    noise = torch.randn_like(Ycf) * max(0.0, 0.2 * (1 - epoch / epochs))
                    Ycf = Ycf + noise
                    Yb = Yb + noise
            real_logits = model.discriminator(hb, Yb, Tb)
            fake_logits = model.discriminator(hb, Ycf, Tb)
            if use_wgan_gp:
                wdist = fake_logits.mean() - real_logits.mean()
                gp = 0.0
                if lambda_gp > 0:
                    eps = torch.rand_like(Yb)
                    interpolates = eps * Yb + (1 - eps) * Ycf
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
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
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
                t_logits = model.discriminator(torch.cat([grad_reverse(hb, grl_weight), Yb, Tb], 1))
                loss_grl = bce(t_logits, Tb)
                loss_g += loss_grl

            opt_g.zero_grad(); loss_g.backward();
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
        history.append(stats)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(
                f"epoch {epoch:2d}",
                f"Ly={stats.loss_y:.3f}",
                f"Lcons={stats.loss_cons:.3f}",
                f"Ladv={stats.loss_adv:.3f}",
            )

    model.eval()
    return (model, history) if return_history else model
