import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
from . import register_operator, LinearOperator, LinearSVDOperator, NonLinearOperator

@register_operator(name='source_separation')
class SourceSeparation(NonLinearOperator):
    def __init__(self, channels, device):
        self.channels = channels
        self.device = device

    def forward(self, x): # x_1 + x_2
        # x: (B, C, T) -> (1, C, T)
        return x.sum(dim=0, keepdim=True)

    def proximal_generator(self, x, y, diffusion, i, sigma, rho, gamma=1e-4):
        z = x.clone().detach()
        # alpha = 0.5
        n_spk = z.shape[0]
        log_p_y_x = (y - (
            torch.stack(torch.chunk(z, n_spk, 0)).sum(0)
        ))
        log_p_y_x = (repeat(log_p_y_x, "h ... -> (r h) ...", r=n_spk))
        z = z + log_p_y_x / n_spk
        t = torch.tensor([i] * z.shape[0])
        # print(log_p_y_x.sum(dim=-1, keepdim=True))
        # print(t)
        if t[0] != 0:
            z = diffusion.q_sample(z, t)
        # z = F.normalize(z)
        # z.requires_grad_(True)
        # print(torch.max(z))
        
        # for _ in range(num_iters):
        #     data_fit = (self.forward(z) - y).norm()**2 / (2* sigma**2)
        #     grad = torch.autograd.grad(outputs=data_fit, inputs=z)[0]
        #     # z = z - gamma * grad - (gamma / rho**2) * (z - x) + np.sqrt(2 * gamma) * torch.randn_like(x)
        #     z = z - gamma * grad + np.sqrt(2 * gamma) * torch.randn_like(x)

        return z.float()

    # def initialize(self, gt, y):
    #     torch.randn_like(gt)

    # def __init__(self, channels, ratio, device):
    #     self.channels = channels
    #     self.ratio = ratio # ratio = 2
    #     A = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
    #     self.U_small, self.singulars_small, self.V_small = torch.svd(A, some=False)
    #     self.Vt_small = self.V_small.transpose(0, 1) # transpose matrix

    # def V(self, signal):
    #     """
    #     還原語音訊號
    #     signal: (batch, 1, T_reduced)
    #     return: (batch, 1, T)
    #     """
    #     B, C, T_reduced = signal.shape
    #     assert C == 1

    #     signal = signal.reshape(B, C, T_reduced // self.ratio, self.ratio)
    #     print(signal)
    #     # use V_small to restore
    #     restored = torch.matmul(self.V_small, signal.unsqueeze(-1)).squeeze(-1)
    #     # print(restored.shape)

    #     return restored.reshape(B, C, -1) # (batch, 1, T)

    # def Vt(self, signal):
    #     """
    #     將語音訊號沿著時間維度 T 進行降維，類似於影像壓縮時的 `Vt`
    #     signal: (batch, channel, T)
    #     return: (batch, channel, reduced_T)
    #     """
    #     B, C, T = signal.shape
    #     assert C == 1
    #     assert T % self.ratio == 0 # T 可以被整除

    #     signal = signal.reshape(B, C, T // self.ratio, self.ratio)
    #     compressed = torch.matmul(self.Vt_small, signal.unsqueeze(-1))
        
    #     return compressed.reshape(B, C, -1)

    # def U(self, signal):
    #     return self.U_small[0, 0] * signal.reshape(signal.shape[0], -1)

    # def Ut(self, signal):
    #     return self.U(signal)

    # def proximal_generator(self, x, y, sigma, rho):
    #     """
    #     x: (batch, 1, T)
    #     y: (batch, 1, T)
    #     """
    #     singulars = torch.ones_like(x)
    #     Qx_inv_eigvals = 1 / (singulars**2 / sigma**2 + 1 / rho**2)
    #     noise = self.V(torch.sqrt(Qx_inv_eigvals) * torch.randn_like(x))
    #     mu_x = self.V(Qx_inv_eigvals * self.Vt(y / sigma**2 + x / rho**2))

    #     return mu_x + noise
