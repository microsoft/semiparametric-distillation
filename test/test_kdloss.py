from pathlib import Path
import sys
sys.path.insert(0, Path(__file__).parent.parent.absolute())
print(sys.path)

import math
import unittest

import torch
import torch.nn.functional as F

from kd import KDOrthoLoss


class KDLossTest(unittest.TestCase):

    def kd_ortho_loss_test(self):

        def cross_entropy(logit_q, p, temperature=1.0):
            log_q = F.log_softmax(logit_q / temperature, dim=-1)
            log_p = torch.log(p)
            w = F.softmax(log_p / temperature, dim=-1)
            return -(w * log_q).sum(dim=-1).mean(dim=0)

        def cross_entropy_temp(student_logit, p, temperature=1.0):
            log_p = torch.log(p)
            w = F.softmax(log_p / temperature, dim=-1)
            return (torch.logsumexp(student_logit / self.temperature, dim=-1)
                    - (w * student_logit).sum(dim=-1) / self.temperature).mean(dim=0)

        def cross_entropy_grad(logit_q, p, temperature=1.0):
            p.requires_grad = True
            ce = cross_entropy(logit_q, p, temperature)
            grad, = torch.autograd.grad(ce, p)
            return grad

        batch_size = 3
        nclasses = 10
        temperature = 4.0
        student_logit = torch.randn(batch_size, nclasses, dtype=torch.float32) * 3
        teacher_logit = torch.randn(batch_size, nclasses, dtype=torch.float32) * 3
        target = torch.randint(0, nclasses, (batch_size, ))

        p = F.softmax(teacher_logit, dim=-1)
        logit_q = student_logit
        log_q = F.log_softmax(logit_q / temperature, dim=-1)
        ce = cross_entropy(logit_q, p, temperature)
        ce_alt = -((p ** (1.0 / temperature) / (p ** (1.0 / temperature)).sum(dim=-1, keepdim=True)) * log_q).sum(dim=-1).mean()
        ce_temp = cross_entropy_temp(logit_q, p, temperature)
        self.assertTrue(torch.allclose(ce, ce_alt))

        w = F.softmax(teacher_logit / temperature, dim=-1)
        kl = F.kl_div(log_q, w, reduction='batchmean')
        entropy = -(w * torch.log(w)).sum(dim=-1).mean()
        self.assertTrue(torch.allclose(ce, kl + entropy))

        grad = cross_entropy_grad(logit_q, p, temperature)
        first_order_term = ((F.one_hot(target, nclasses).float() - p) * grad).sum(dim=-1).sum(dim=0)
        kd_ortho_loss = KDOrthoLoss(temperature, alpha=0.5)
        first_order_term_kd = kd_ortho_loss._first_order_term(student_logit, teacher_logit, target)
        self.assertTrue(torch.allclose(first_order_term, first_order_term_kd),
                        (first_order_term - first_order_term_kd).item())
        print(kl, first_order_term)
