import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Implements ASL with adaptive negative focusing.
class AsymmetricGBLoss(nn.Module):
    """
    Asymmetric Loss with gradient-budget constrained adaptation for gamma_neg.

    Baseline loss form source:
    - Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
      CVF page: https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html
      PDF: https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf

    This implementation removes the original Eq.11-style delta-p heuristic update
    (gamma_minus <- gamma_minus + lambda * (Delta p - Delta p_target)) and replaces
    it with a constrained 1D optimization on gradient budgets:

        gamma_t* = argmin_{gamma in [gmin, gmax]}
            ( log(G_t^-(gamma) + eps) - log(rho * G_t^+ + eps) )^2
            + beta * (gamma - gamma_{t-1})^2

    where:
    - G_t^+ is average positive gradient magnitude proxy.
    - G_t^-(gamma) is average negative gradient magnitude proxy under gamma.
    - rho is target negative/positive gradient ratio.
    - beta is temporal regularization for smooth gamma trajectory.

    Related theoretical context:
    - Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
      https://arxiv.org/abs/1708.02002
    - Li et al., "Gradient Harmonized Single-stage Detector", AAAI 2019.
      https://arxiv.org/abs/1811.05181
    - Mukhoti et al., "AdaFocal", NeurIPS 2022.
      https://openreview.net/forum?id=kUOm0Fdtvh
    """

    def __init__(
            self,
            gamma_neg=4.0,
            gamma_pos=1.5,
            clip=0.05,
            eps=1e-6,
            gamma_neg_min=3.0,
            gamma_neg_max=6.0,
            gamma_balance_ratio=0.9,
            gamma_reg_beta=0.15,
            gamma_update_lr=0.05,
            gamma_update_iters=1,
            gamma_fd_eps=0.02,
    ):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.clip = clip
        self.eps = float(eps)

        # Gradient-budget adaptation hyper-parameters.
        self.gamma_neg_min = float(gamma_neg_min)
        self.gamma_neg_max = float(gamma_neg_max)
        self.gamma_balance_ratio = float(gamma_balance_ratio)
        self.gamma_reg_beta = float(gamma_reg_beta)
        self.gamma_update_lr = float(gamma_update_lr)
        self.gamma_update_iters = int(gamma_update_iters)
        self.gamma_fd_eps = float(gamma_fd_eps)

        if self.gamma_neg_min > self.gamma_neg_max:
            raise ValueError("gamma_neg_min must be <= gamma_neg_max.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")

        # Keep the initial value inside the feasible update range.
        self.gamma_neg = float(min(self.gamma_neg_max, max(self.gamma_neg_min, gamma_neg)))

        self.last_pos_loss = None
        self.last_neg_loss = None
        self.last_grad_pos = None
        self.last_grad_neg = None
        self.last_grad_ratio = None

    def _negative_margin(self, p):
        """Apply ASL clipping on negative probabilities before focal reweighting."""

        if self.clip is not None and self.clip > 0:
            neg_margin = (p - self.clip).clamp(min=0.0)
        else:
            neg_margin = p
        return neg_margin.clamp(max=1.0 - self.eps)

    def _mean_abs_pos_grad_proxy(self, p, y):
        """
        Positive gradient proxy for -y * (1-p)^gamma_pos * log(p).

        We use:
            d(-logsigmoid(x))/dx = 1 - sigmoid(x) = 1 - p
        and keep the focal multiplicative term (1-p)^gamma_pos.
        """
        pos_mass = y.sum()
        if pos_mass <= self.eps:
            return 0.0

        if self.gamma_pos > 0:
            pos_w = (1.0 - p).pow(self.gamma_pos)
        else:
            pos_w = torch.ones_like(p)

        grad_mag = y * pos_w * (1.0 - p)
        return float(grad_mag.sum() / (pos_mass + self.eps))

    def _mean_abs_neg_grad_proxy(self, p, y, neg_margin, gamma_neg):
        """
        Negative gradient proxy for -(1-y) * m^gamma * log(1-m), m in [0, 1).

        Let f(m) = -m^gamma * log(1-m). Then:
            df/dm = gamma * m^(gamma-1) * (-log(1-m)) + m^gamma / (1-m)
        and the chain term dm/dx is applied.

        Here m = max(p - clip, 0) (or p when clip<=0), so dm/dx includes an
        active mask for m>0 and sigmoid slope p*(1-p).
        """
        neg_mask = (1.0 - y)
        neg_mass = neg_mask.sum()
        if neg_mass <= self.eps:
            return 0.0

        active = (neg_margin > 0).to(dtype=p.dtype)
        dm_dx = p * (1.0 - p) * active

        if gamma_neg <= 0:
            df_dm = 1.0 / (1.0 - neg_margin + self.eps)
        else:
            minus_log_one_minus_m = -torch.log1p(-neg_margin)
            safe_m = neg_margin.clamp(min=self.eps)
            left = gamma_neg * safe_m.pow(gamma_neg - 1.0) * minus_log_one_minus_m
            right = neg_margin.pow(gamma_neg) / (1.0 - neg_margin + self.eps)
            df_dm = left + right

        grad_mag = neg_mask * df_dm * dm_dx
        return float(grad_mag.sum() / (neg_mass + self.eps))

    def _update_gamma_neg_by_gradient_budget(self, p, y):
        """
        Update gamma_neg by solving a 1D constrained objective each batch.

        Objective:
            min_gamma [ log(G^-(gamma)+eps) - log(rho*G^+ + eps) ]^2
                      + beta * (gamma - gamma_prev)^2

        Difference from ASL Eq.11:
        - ASL Eq.11 updates gamma_neg using probability-gap Delta p.
        - This implementation uses direct gradient-budget matching, which is a
          different mechanism and targets gradient flow balance explicitly.
        """
        if self.gamma_update_iters <= 0:
            return

        with torch.no_grad():
            pos_mass = y.sum()
            neg_mass = (1.0 - y).sum()
            if pos_mass <= self.eps or neg_mass <= self.eps:
                return

            p_detached = p.detach()
            y_detached = y.detach()
            neg_margin = self._negative_margin(p_detached)

            g_pos = self._mean_abs_pos_grad_proxy(p_detached, y_detached)
            target_log = math.log(self.gamma_balance_ratio * g_pos + self.eps)
            prev_gamma = float(min(self.gamma_neg_max, max(self.gamma_neg_min, self.gamma_neg)))

            def objective(gamma_value):
                g_neg = self._mean_abs_neg_grad_proxy(
                    p_detached, y_detached, neg_margin, gamma_value
                )
                diff = math.log(g_neg + self.eps) - target_log
                reg = self.gamma_reg_beta * (gamma_value - prev_gamma) ** 2
                return diff * diff + reg

            gamma = prev_gamma
            for _ in range(self.gamma_update_iters):
                h = max(self.gamma_fd_eps, self.eps)
                g_up = min(self.gamma_neg_max, gamma + h)
                g_dn = max(self.gamma_neg_min, gamma - h)
                if g_up <= g_dn:
                    break

                obj_up = objective(g_up)
                obj_dn = objective(g_dn)
                grad = (obj_up - obj_dn) / (g_up - g_dn + self.eps)

                gamma = gamma - self.gamma_update_lr * grad
                gamma = float(min(self.gamma_neg_max, max(self.gamma_neg_min, gamma)))

            self.gamma_neg = gamma
            g_neg_final = self._mean_abs_neg_grad_proxy(
                p_detached, y_detached, neg_margin, self.gamma_neg
            )
            self.last_grad_pos = float(g_pos)
            self.last_grad_neg = float(g_neg_final)
            self.last_grad_ratio = float(g_neg_final / (g_pos + self.eps))

    def forward(self, x, y):
        """Compute the asymmetric loss for a batch of logits and binary targets."""

        y = y.to(dtype=x.dtype)

        # Convert logits to probabilities once because both branches and the
        # adaptive gamma update use the same sigmoid output.
        p = torch.sigmoid(x)
        # The adaptive state must stay frozen during validation/testing.
        if self.training and torch.is_grad_enabled():
            self._update_gamma_neg_by_gradient_budget(p, y)

        # Positive branch: emphasize hard positive examples.
        pos_log = F.logsigmoid(x)
        if self.gamma_pos > 0:
            pos_w = (1.0 - p).pow(self.gamma_pos)
            pos_log = pos_log * pos_w
        los_pos = y * pos_log

        # Negative branch: clip easy negatives, then apply asymmetric focusing.
        neg_margin = self._negative_margin(p)
        neg_log = torch.log1p(-neg_margin)

        if self.gamma_neg > 0:
            neg_w = neg_margin.pow(self.gamma_neg)
            neg_log = neg_log * neg_w

        los_neg = (1.0 - y) * neg_log

        # Store detached diagnostics so the training loop can inspect the
        # positive and negative contribution balance without retaining graphs.
        self.last_pos_loss = (-los_pos.sum()).detach()
        self.last_neg_loss = (-los_neg.sum()).detach()

        return (-(los_pos + los_neg).sum(dim=-1)).mean()


# https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
# Implements the reference asymmetric multilabel loss.
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1.5, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


# Implements a memory-conscious asymmetric multilabel loss.
class AsymmetricLossOptimized(nn.Module):
    """
    Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    """

    def __init__(self, gamma_neg=4, gamma_pos=1.5, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


# Implements ASL for single-label classification.
class ASLSingleLabel(nn.Module):
    """
    This loss is intended for single-label classification problems
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
