import torch
from torch import nn
from torch.nn import functional as F


class KDLoss(nn.Module):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, temperature=1.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logit, teacher_logit, target, loss_original):
        # Adapted from https://github.com/huggingface/pytorch-transformers/blob/master/examples/distillation/distiller.py
        # Scaled by temperature^2 to balance the soft and hard loss
        # See https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
        # or https://github.com/stanford-futuredata/lit-code/blob/master/cifar10/distillation_loss.py
        kl = F.kl_div(F.log_softmax(student_logit / self.temperature, dim=-1),
                      F.softmax(teacher_logit / self.temperature, dim=-1), reduction='batchmean')
        loss_kd = kl * self.temperature ** 2
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


class KDMSELoss(nn.Module):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, alpha=0.5, scale='logp'):
        super().__init__()
        assert scale in ['logp', 'p']
        self.alpha = alpha
        self.scale = scale

    def forward(self, student_logit, teacher_logit, target, loss_original):
        if self.scale == 'logp':
            loss_kd = F.mse_loss(student_logit, teacher_logit)
        else:
            p = F.softmax(teacher_logit, dim=-1)
            loss_kd = F.mse_loss(student_logit, p)
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


class KDOrthoLoss(KDLoss):
    """
    Orthogonal loss with knowledge distillation.
    """
    def __init__(self, temperature=1.0, alpha=0.5, eps=1e-2, smoothing=0.0):
        super().__init__(temperature, alpha)
        self.eps = eps
        self.smoothing = smoothing  # How much to shrink toward uniform

    def forward(self, student_logit, teacher_logit, target, loss_original):
        kl = F.kl_div(F.log_softmax(student_logit / self.temperature, dim=-1),
                      F.softmax(teacher_logit / self.temperature, dim=-1), reduction='batchmean')
        # w = F.softmax(teacher_logit / self.temperature, dim=-1)
        # ce_temp = (torch.logsumexp(student_logit / self.temperature, dim=-1)
        #            - (w * student_logit).sum(dim=-1) / self.temperature).mean()
        # p = F.softmax(teacher_logit, dim=-1)
        # logit_q = student_logit
        # ce = cross_entropy(logit_q, p, self.temperature)
        # log_q = F.log_softmax(logit_q / temperature, dim=-1)
        # psi = lambda s: torch.logsumexp(s / self.temperature, dim=-1)
        first_order_term = self._first_order_term(student_logit, teacher_logit, target)
        loss_kd = (kl + first_order_term) * self.temperature ** 2
        # print(loss_original.item(), kl.item(), first_order_term.item())
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd

    def _first_order_term(self, student_logit, teacher_logit, target):
        a = 1.0 / self.temperature
        w = F.softmax(teacher_logit / self.temperature, dim=-1)
        log_q = F.log_softmax(student_logit / self.temperature, dim=-1)
        p = F.softmax(teacher_logit, dim=-1)
        if self.smoothing > 0.0:
            uniform = torch.ones_like(p) / student_logit.shape[-1]
            p = (1.0 - self.smoothing) * p + self.smoothing * uniform
        y = F.one_hot(target, student_logit.shape[-1]).float()
        yp_1 = y / p.clamp(self.eps) - 1.0
        # print(yp_1.max().item(), (w / p).max().item())
        yp_1_w = yp_1 * w
        # print(yp_1_w.max().item(), yp_1_w.abs().mean().item())
        # TODO: einsum or tensordot is probably faster
        ortho1 = (yp_1_w.sum(dim=-1) * (w * log_q).sum(dim=-1))
        ortho2 = -(yp_1_w * log_q).sum(dim=-1)
        # y_p = (F.one_hot(target, nclasses).float() - p)
        # ortho1 = a * ((y_p * w).sum(dim=-1) * (w * log_q / p).sum(dim=-1)).mean()
        # ortho2 = -a * (y_p * w * log_q / p).sum(dim=-1).mean()

        # ortho1 = a * ((y_p * (w/p)).sum(dim=-1) * (w * log_q).sum(dim=-1))
        # ortho2 = -a * (y_p * w * log_q / p).sum(dim=-1)
        return a * (ortho1 + ortho2).mean(dim=0)

        # temp1 = (yp_1_w.sum(dim=-1) * (w * student_logit).sum(dim=-1))
        # temp2 = -(yp_1_w * student_logit).sum(dim=-1)
        # return a**2 * (temp1 + temp2).mean(dim=0)

        # yp = y / p.clamp(self.eps)
        # yp_w = yp * w
        # temp1 = (yp_w.sum(dim=-1) * (w * student_logit).sum(dim=-1))
        # temp2 = -(yp_w * student_logit).sum(dim=-1)
        # return a**2 * (temp1 + temp2).mean(dim=0)


class KDMSEOrthoLoss(KDMSELoss):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, alpha=0.5, eps=1e-2, smoothing=0.0):
        super().__init__(alpha)
        self.eps = eps
        self.smoothing = smoothing

    def forward(self, student_logit, teacher_logit, target, loss_original):
        p = F.softmax(teacher_logit, dim=-1)
        if self.smoothing > 0.0:
            uniform = torch.ones_like(p) / student_logit.shape[-1]
            p = (1.0 - self.smoothing) * p + self.smoothing * uniform
        y = F.one_hot(target, student_logit.shape[-1]).float()
        yp_1 = y / p.clamp(self.eps) - 1.0
        loss_kd = F.mse_loss(student_logit, teacher_logit + yp_1)
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


def find_optimal_gamma_sampling(phat, bound_fn, max_range=10, alpha=1.0, scale='logp'):
    assert scale in ['p', 'logp']
    phat_shape = phat.shape
    phat = phat.flatten()
    gamma = torch.arange(-max_range, max_range, 0.05, device=phat.device).unsqueeze(-1).unsqueeze(-1)
    if scale == 'p':
        def objective(p):
            return (gamma * (p - phat) - (p - phat))**2 + (1 / alpha - 1 + gamma)**2 * p * (1 - p)
    else:
        def objective(p):
            return (gamma * (p - phat) - (torch.log(p) - torch.log(phat)))**2 + (1 / alpha - 1 + gamma)**2 * p * (1 - p)
    bound_l, bound_h = bound_fn(phat)
    p_vals = bound_l + torch.linspace(0.0, 1.0, 10, device=phat.device).unsqueeze(-1) * (bound_h - bound_l)
    objs = objective(p_vals)
    max_objs = objs.max(dim=1)[0]
    return gamma[torch.argmin(max_objs, dim=0)].reshape(*phat_shape)


def find_optimal_gamma_relerr(phat, c, max_range=10, alpha=1.0):
    phat_shape = phat.shape
    phat = phat.flatten()
    gamma = torch.arange(-max_range, max_range, 0.05, device=phat.device).unsqueeze(-1).unsqueeze(-1)
    def objective(p):
        return (gamma * (p - phat) - (torch.log(p) - torch.log(phat)))**2 + (1 / alpha - 1 + gamma)**2 * p * (1 - p)
    bound_l = torch.clamp(phat / (1 + c), min=1e-6)
    bound_h = torch.clamp(phat * (1 + c), max=1.0)
    p_vals = bound_l + torch.linspace(0.0, 1.0, 10, device=phat.device).unsqueeze(-1) * (bound_h - bound_l)
    objs = objective(p_vals)
    max_objs = objs.max(dim=1)[0]
    return gamma[torch.argmin(max_objs, dim=0)].reshape(*phat_shape)


class KDMSEMinimaxRelerrLoss(KDMSELoss):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, alpha=0.5, scale='logp', smoothing=0.0, c=2.0):
        super().__init__(alpha, scale)
        self.smoothing = smoothing
        self.c = c

    def forward(self, student_logit, teacher_logit, target, loss_original):
        p = F.softmax(teacher_logit, dim=-1)
        if self.smoothing > 0.0:
            uniform = torch.ones_like(p) / student_logit.shape[-1]
            p = (1.0 - self.smoothing) * p + self.smoothing * uniform
        # Convert student_logit and teacher_logit to log probabilities
        # Nvm this makes the loss NaN
        # student_logit = F.log_softmax(student_logit, dim=-1)
        # teacher_logit = F.log_softmax(teacher_logit, dim=-1)
        y = F.one_hot(target, student_logit.shape[-1]).float()
        bound_l = 0.0 if self.scale == 'p' else 1e-6
        bound_fn = lambda phat: (torch.clamp(phat / (1 + self.c), min=bound_l),
                            torch.clamp(phat * (1 + self.c), max=1.0))
        gamma = find_optimal_gamma_sampling(p, bound_fn, alpha=self.alpha, scale=self.scale)
        if self.scale == 'logp':
            loss_kd = F.mse_loss(student_logit, teacher_logit + gamma * (y - p))
        else:
            loss_kd = F.mse_loss(student_logit, p + gamma * (y - p))
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


class KDMSEGamma1Loss(KDMSELoss):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, alpha=0.5, smoothing=0.0):
        super().__init__(alpha)
        self.smoothing = smoothing

    def forward(self, student_logit, teacher_logit, target, loss_original):
        p = F.softmax(teacher_logit, dim=-1)
        if self.smoothing > 0.0:
            uniform = torch.ones_like(p) / student_logit.shape[-1]
            p = (1.0 - self.smoothing) * p + self.smoothing * uniform
        y = F.one_hot(target, student_logit.shape[-1]).float()
        gamma = 1.0
        loss_kd = F.mse_loss(student_logit, teacher_logit + gamma * (y - p))
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


class KDMSEGammaVarLoss(KDMSELoss):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, alpha=0.5, smoothing=0.0):
        super().__init__(alpha)
        self.smoothing = smoothing

    def forward(self, student_logit, teacher_logit, target, loss_original):
        p = F.softmax(teacher_logit, dim=-1)
        if self.smoothing > 0.0:
            uniform = torch.ones_like(p) / student_logit.shape[-1]
            p = (1.0 - self.smoothing) * p + self.smoothing * uniform
        y = F.one_hot(target, student_logit.shape[-1]).float()
        gamma = 1 - p
        loss_kd = F.mse_loss(student_logit, teacher_logit + gamma * (y - p))
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


def find_optimal_gamma_power(phat, tmax, max_range=10, alpha=1.0):
    phat_shape = phat.shape
    phat = phat.flatten()
    gamma = torch.arange(-max_range, max_range, 0.05, device=phat.device).unsqueeze(-1).unsqueeze(-1)
    def objective(p):
        return (gamma * (p - phat) - (torch.log(p) - torch.log(phat)))**2 + (1 / alpha - 1 + gamma)**2 * p * (1 - p)
    bound_l = torch.clamp(phat, min=1e-6)
    bound_h = torch.clamp(phat ** (1 / tmax), max=1.0)
    p_vals = bound_l + torch.linspace(0.0, 1.0, 10, device=phat.device).unsqueeze(-1) * (bound_h - bound_l)
    objs = objective(p_vals)
    max_objs = objs.max(dim=1)[0]
    return gamma[torch.argmin(max_objs, dim=0)].reshape(*phat_shape)


class KDMSEMinimaxPowerLoss(KDMSELoss):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, alpha=0.5, scale='logp', smoothing=0.0, tmax=2.0):
        super().__init__(alpha, scale)
        self.smoothing = smoothing
        self.tmax = tmax

    def forward(self, student_logit, teacher_logit, target, loss_original):
        p = F.softmax(teacher_logit, dim=-1)
        if self.smoothing > 0.0:
            uniform = torch.ones_like(p) / student_logit.shape[-1]
            p = (1.0 - self.smoothing) * p + self.smoothing * uniform
        y = F.one_hot(target, student_logit.shape[-1]).float()
        bound_l = 0.0 if self.scale == 'p' else 1e-6
        bound_fn = lambda phat: (torch.clamp(phat, min=bound_l),
                            torch.clamp(phat ** (1 / self.tmax), max=1.0))
        gamma = find_optimal_gamma_sampling(p, bound_fn, alpha=self.alpha, scale=self.scale)
        if self.scale == 'logp':
            loss_kd = F.mse_loss(student_logit, teacher_logit + gamma * (y - p))
        else:
            loss_kd = F.mse_loss(student_logit, p + gamma * (y - p))
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


def find_optimal_gamma_abserr(phat, c, max_range=10, alpha=1.0):
    phat_shape = phat.shape
    phat = phat.flatten()
    gamma = torch.arange(-max_range, max_range, 0.05, device=phat.device).unsqueeze(-1).unsqueeze(-1)
    def objective(p):
        return (gamma * (p - phat) - (torch.log(p) - torch.log(phat)))**2 + (1 / alpha - 1 + gamma)**2 * p * (1 - p)
    bound_l = torch.clamp(phat - c, min=1e-3)
    bound_h = torch.clamp(phat + c, max=1.0)
    p_vals = bound_l + torch.linspace(0.0, 1.0, 10, device=phat.device).unsqueeze(-1) * (bound_h - bound_l)
    objs = objective(p_vals)
    max_objs = objs.max(dim=1)[0]
    return gamma[torch.argmin(max_objs, dim=0)].reshape(*phat_shape)


class KDMSEMinimaxAbserrLoss(KDMSELoss):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, alpha=0.5, scale='logp', smoothing=0.0, c=0.05):
        super().__init__(alpha, scale)
        self.smoothing = smoothing
        self.c = c

    def forward(self, student_logit, teacher_logit, target, loss_original):
        p = F.softmax(teacher_logit, dim=-1)
        if self.smoothing > 0.0:
            uniform = torch.ones_like(p) / student_logit.shape[-1]
            p = (1.0 - self.smoothing) * p + self.smoothing * uniform
        y = F.one_hot(target, student_logit.shape[-1]).float()
        bound_l = 0.0 if self.scale == 'p' else 1e-3
        bound_fn = lambda phat: (torch.clamp(phat - self.c, min=bound_l),
                            torch.clamp(phat + self.c, max=1.0))
        gamma = find_optimal_gamma_sampling(p, bound_fn, alpha=self.alpha, scale=self.scale)
        if self.scale == 'logp':
            loss_kd = F.mse_loss(student_logit, teacher_logit + gamma * (y - p))
        else:
            loss_kd = F.mse_loss(student_logit, p + gamma * (y - p))
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


class KDMSEBoundFast(KDMSELoss):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, alpha=0.5, scale='logp', smoothing=0.0, c=1.0):
        super().__init__(alpha, scale)
        self.smoothing = smoothing
        self.c = c

    def forward(self, student_logit, teacher_logit, target, loss_original):
        p = F.softmax(teacher_logit, dim=-1)
        if self.smoothing > 0.0:
            uniform = torch.ones_like(p) / student_logit.shape[-1]
            p = (1.0 - self.smoothing) * p + self.smoothing * uniform
        y = F.one_hot(target, student_logit.shape[-1]).float()
        bound_l = 0.0 if self.scale == 'p' else 1e-3
        p_clipped = torch.clamp(p, min=bound_l)
        if self.scale == 'logp':
            gamma = self.c / p_clipped / (self.c + (y - p)**2)
            loss_kd = F.mse_loss(student_logit, teacher_logit + gamma * (y - p))
        else:
            gamma = self.c / (self.c + (y - p)**2)
            loss_kd = F.mse_loss(student_logit, p + gamma * (y - p))
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


class KDMSEVarRedOrthoLoss(KDMSELoss):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, alpha=0.5, smoothing=0.0):
        super().__init__(alpha)
        self.smoothing = smoothing

    def forward(self, student_logit, teacher_logit, target, loss_original):
        p = F.softmax(teacher_logit, dim=-1)
        if self.smoothing > 0.0:
            uniform = torch.ones_like(p) / student_logit.shape[-1]
            p = (1.0 - self.smoothing) * p + self.smoothing * uniform
        y = F.one_hot(target, student_logit.shape[-1]).float()
        yp_1 = y / p - 1.0
        # TODO Cross term to avoid numerical error
        # TODO Try explicit gradient
        loss_kd = ((student_logit - (teacher_logit + yp_1))**2 * p).mean()
        temp = ((student_logit - teacher_logit - y / p + 1.0)**2 * p).mean()
        temp = (((student_logit - teacher_logit + 1.0)**2 * p + y**2/p - 2*y*(student_logit - teacher_logit + 1.0)) ).mean()
        temp = (((student_logit - teacher_logit)**2 * p + (y-p)**2/p - 2*(y-p)*(student_logit - teacher_logit)) ).mean()
        return (1 - self.alpha) * loss_original + self.alpha * loss_kd


