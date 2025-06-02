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
        x = x.to(self.device)
        y = y.to(self.device)
        # z = x.clone().detach()
        # n_spk = z.shape[0]
        batch_nspk, C, T = x.shape
        n_spk = self.channels
        batch_size = batch_nspk // n_spk
        assert batch_nspk == batch_size * n_spk, "batch size 不整除 speaker 數"

        z = x.clone().detach().view(batch_size, n_spk, C, T)
        y = y.view(batch_size, C, T)  # y 是 sum 後的混合訊號，只有 batch_size 個
        t = torch.tensor([i] * batch_size, device=self.device)
        print(f"z.shape: {z.shape}, y.shape: {y.shape}")
        # === Add orthogonality regularization ===
        z.requires_grad_(True)
        for _ in range(3):
            z_reshaped = z.view(batch_size * n_spk, C, T).squeeze(1)  # (B * n_spk, T)
            embedding = classifier.encode_batch(z_reshaped.to(self.device))
            embedding = embedding.view(batch_size, n_spk, -1)  # (B, n_spk, D)
            ortho_loss = self.compute_ortho_loss(embedding.squeeze(1))
            ortho_loss = ortho_loss.mean()  # 平均化損失
            adaptive_rho = self.compute_adaptive_rho(rho, i, ortho_loss.detach())
            grad = torch.autograd.grad(ortho_loss, z, retain_graph=True)[0].detach()
            z = z - (adaptive_rho * gamma) * grad
            print(adaptive_rho, adaptive_rho * gamma)

            # log_p_y_x = (y - (
            #     torch.stack(torch.chunk(z, n_spk, 0)).sum(0)
            # ))
            # log_p_y_x = repeat(log_p_y_x, "h ... -> (r h) ...", r=n_spk) / n_spk
            log_p_y_x = y - z.sum(dim=1)
            log_p_y_x = log_p_y_x.unsqueeze(1).repeat(1, n_spk, 1, 1) / n_spk
            z = z + log_p_y_x

        z = z - (gamma / rho**2) * (z - x.view(batch_size, n_spk, C, T))  # proximal step
        # print(f"rho: {rho}")
        if t[0] != 0:
            z_flat = z.view(batch_size * n_spk, C, T)
            z_flat = diffusion.q_sample(z_flat, t.repeat_interleave(n_spk))
            z = z_flat.view(batch_size, n_spk, C, T)
            # z = diffusion.q_sample(z, t)   

        # return z.float()
        return z.view(batch_size * n_spk, C, T).float()  # (B * n_spk, C, T)

    # def compute_ortho_loss(self, z_):
    #     z_norm = F.normalize(z_, p=2, dim=-1)
    #     cos_matrix = torch.matmul(z_norm, z_norm.T) # z_norm @ z_norm^T
    #     return (cos_matrix.abs().sum() - z_.shape[0]) / (z_.shape[0] * (z_.shape[0] - 1))
        
    def compute_ortho_loss(self, z_):
        """
        根據論文公式 ||V^T V - I||_F^2 實作 z_ 為 [B, D] 的嵌入矩陣。
        """
        # z_norm = F.normalize(z_, p=2, dim=-1)  # 每個 embedding 單位化
        # dot_matrix = torch.matmul(z_norm.T, z_norm)  # V^T V, shape = [D, D]
        # identity = torch.eye(dot_matrix.size(0), device=z_.device)
        # return F.mse_loss(dot_matrix, identity)  # 等價於 Frobenius norm 平方
        B, n_spk, D = z_.shape
        z_norm = F.normalize(z_, p=2, dim=-1)  # [B, n_spk, D]
        loss = 0
        for b in range(B):
            mat = torch.matmul(z_norm[b], z_norm[b].T)  # [n_spk, n_spk]
            identity = torch.eye(n_spk, device=z_.device)
            loss += F.mse_loss(mat, identity)
        return loss / B
    
    def compute_adaptive_rho(self, rho, i, loss_value):
        """自適應調整 rho 參數"""
        decay_factor = 0.98 ** (i / 10)
        loss_factor = 1.0 + torch.sigmoid(5 * (loss_value - 0.1))
        return rho * decay_factor * loss_factor
   
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
