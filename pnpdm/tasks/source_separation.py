import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
from . import register_operator, LinearOperator, LinearSVDOperator, NonLinearOperator
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


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

        log_p_y_x = (y - (
            torch.stack(torch.chunk(z, n_spk, 0)).sum(0)
        ))
        log_p_y_x = repeat(log_p_y_x, "h ... -> (r h) ...", r=n_spk) / n_spk
        z = z + log_p_y_x

        # === Add orthogonality regularization ===
        z.requires_grad_(True)
        # for _ in range(5):
        embedding = classifier.encode_batch(z.squeeze(1))

        ortho_loss = self.compute_ortho_loss(embedding.squeeze(1))
        grad = torch.autograd.grad(ortho_loss, z, retain_graph=True)[0].detach()
        z = z - (gamma / rho**2) * grad

        # print(f"rho: {rho}")
        if t[0] != 0:
            z = diffusion.q_sample(z, t)
            z = z - (gamma / rho**2) * (z - x)

        return z.float()

    def compute_ortho_loss(self, z_):
        z_norm = F.normalize(z_, p=2, dim=-1)
        cos_matrix = torch.matmul(z_norm, z_norm.T) # z_norm @ z_norm^T
        return (cos_matrix.abs().sum() - z_.shape[0]) / (z_.shape[0] * (z_.shape[0] - 1))
        
    # def coefficient_cal(self, x, y):
    #     """
    #     用最小平方法計算混合語音中每個來源訊號的係數 alpha
    #     """
    #     # x_1, x_2 = x[0].unsqueeze(0), x[1].unsqueeze(0) # (1, 1, T)

    #     # M = torch.Tensor([[torch.mean(x_1 * x_1), torch.mean(x_1 * x_2)],
    #     #                   [torch.mean(x_2 * x_1), torch.mean(x_2 * x_2)]])
        
    #     # N = torch.Tensor([[torch.mean(x_1 * y)], [torch.mean(x_2 * y)]])

    #     # alpha = torch.linalg.solve(M, N).squeeze()
    #     # print(f"alpha: {alpha}")
    #     # return alpha
    #     x_1, x_2 = x[0].unsqueeze(0), x[1].unsqueeze(0)  # (1, 1, T)
    #     best_loss = float('inf')
    #     best_alpha = None
    #     best_shift = (0, 0)
    #     max_shift = 50  # Maximum shift value

    #     for shift1 in range(-max_shift, max_shift + 1, 10):
    #         for shift2 in range(-max_shift, max_shift + 1, 10):
    #             x1_shifted = torch.roll(x_1, shifts=shift1, dims=-1)
    #             x2_shifted = torch.roll(x_2, shifts=shift2, dims=-1)

    #             M = torch.Tensor([
    #                 [torch.mean(x1_shifted * x1_shifted), torch.mean(x1_shifted * x2_shifted)],
    #                 [torch.mean(x2_shifted * x1_shifted), torch.mean(x2_shifted * x2_shifted)]
    #             ])
    #             N = torch.Tensor([
    #                 [torch.mean(x1_shifted * y)],
    #                 [torch.mean(x2_shifted * y)]
    #             ])

    #             try:
    #                 alpha = torch.linalg.solve(M, N).squeeze()
    #                 recon = alpha[0] * x1_shifted + alpha[1] * x2_shifted
    #                 loss = torch.mean((recon - y) ** 2)

    #                 if loss < best_loss:
    #                     best_loss = loss
    #                     best_alpha = alpha
    #                     best_shift = (shift1, shift2)
    #             except RuntimeError:
    #                 continue

    #     print(f"best shift: {best_shift}, alpha: {best_alpha}")
    #     return best_alpha, best_shift
   
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
