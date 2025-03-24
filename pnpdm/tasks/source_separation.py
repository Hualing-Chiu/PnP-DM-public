import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.fft import fft2, ifft2, fftshift
from . import register_operator, LinearOperator, LinearSVDOperator

@register_operator(name='source_separation')
class SourceSeparation(LinearSVDOperator):
    def __init__(self, channels, ratio, device):
        self.channels = channels
        self.ratio = ratio
        A = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(A, some=False)
        self.Vt_small = self.V_small.transpose(0, 1) # transpose matrix

    def V(self, signal):
        """
        還原語音訊號
        signal: (batch, 1, T_reduced)
        return: (batch, 1, T)
        """
        B, C, T_reduced = signal.shape
        assert C == 1

        signal = signal.reshape(B, C, T_reduced // self.ratio, self.ratio)

        # use V_small to restore
        restored = torch.matmul(self.V_small, signal.unsqueeze(-1)).squeeze(-1)

        return restored.reshape(B, C, -1) # (batch, 1, T)

    def Vt(self, signal):
        """
        將語音訊號沿著時間維度 T 進行降維，類似於影像壓縮時的 `Vt`
        signal: (batch, channel, T)
        return: (batch, channel, reduced_T)
        """
        B, C, T = signal.shape
        assert C == 1
        assert T % self.ratio == 0 # T 可以被整除

        signal = signal.reshape(B, C, T // self.ratio, self.ratio)
        compressed = torch.matmul(self.Vt_small, signal.unsqueeze(-1))
        
        return compressed.reshape(B, C, -1)

    def U(self, signal):
        return self.U_small[0, 0] * signal.reshape(signal.shape[0], -1)

    def Ut(self, signal):
        return self.U(signal)

    def proximal_generator(self, x, y, sigma, rho):
        """
        x: (batch, 1, T)
        y: (batch, 1, T)
        """
        singulars = torch.ones_like(x)
        Qx_inv_eigvals = 1 / (singulars**2 / sigma**2 + 1 / rho**2)
        noise = self.V(torch.sqrt(Qx_inv_eigvals) * torch.randn_like(x))
        mu_x = self.V(Qx_inv_eigvals * self.Vt(y / sigma**2 + x / rho**2))

        return mu_x + noise
