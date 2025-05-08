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
        n_spk = z.shape[0]
        t = torch.tensor([i] * z.shape[0])
    
        alpha = self.coefficient_cal(x, y)
        alpha = torch.clamp(alpha, min=1e-4)
        alpha = alpha / (alpha.sum() + 1e-8)
        print(f"alpha: {alpha}")
        alpha = alpha.view(n_spk, 1, 1)
        recon = torch.sum(alpha * z, dim=0, keepdim=True)  # (1, 1, T)
        log_p_y_x = y - recon # (1, 1, T)
        # log_p_y_x = (y - (
        #     torch.stack(torch.chunk(z, n_spk, 0)).sum(0)
        # ))
        log_p_y_x = (repeat(log_p_y_x, "h ... -> (r h) ...", r=n_spk)) / n_spk
        z = z + log_p_y_x / torch.sqrt(alpha)
        # print(log_p_y_x.sum(dim=-1, keepdim=True))
        # print(t)
        if t[0] != 0:
            z = diffusion.q_sample(z, t)
            z = z - (gamma / rho**2) * (z - x)

        return z.float()

    def coefficient_cal(self, x, y):
        """
        用最小平方法計算混合語音中每個來源訊號的係數 alpha
        """
        # x_1, x_2 = x[0].unsqueeze(0), x[1].unsqueeze(0) # (1, 1, T)
        # y_norm = torch.norm(y, dim=-1, keepdim=True) # (1, 1, T)
        # x1_norm = torch.norm(x_1, dim=-1, keepdim=True) # (1, 1, T)
        # x2_norm = torch.norm(x_2, dim=-1, keepdim=True)
        # M = torch.Tensor([[torch.mean(x1_norm * x1_norm), torch.mean(x1_norm * x2_norm)],
        #                   [torch.mean(x1_norm * x2_norm), torch.mean(x2_norm * x2_norm)]])
        
        # N = torch.Tensor([[torch.mean(x1_norm * y_norm)], [torch.mean(x2_norm * y_norm)]])
        
        x_1, x_2 = x[0].unsqueeze(0), x[1].unsqueeze(0) # (1, 1, T)
        y_sq = y ** 2
        x1_sq = x_1 ** 2
        x2_sq = x_2 ** 2
        # print(f"x_1: {x_1.shape}, x_2: {x_2.shape}")
        # print(f"y: {y.shape}")

        M = torch.Tensor([[torch.mean(x1_sq * x1_sq), torch.mean(x1_sq * x2_sq)],
                          [torch.mean(x1_sq * x2_sq), torch.mean(x2_sq * x2_sq)]])
        
        N = torch.Tensor([[torch.mean(x1_sq * y_sq)], [torch.mean(x2_sq * y_sq)]])

        alpha = torch.linalg.solve(M, N).squeeze()
        return alpha
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
