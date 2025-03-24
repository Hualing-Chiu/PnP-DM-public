import torch, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
# from .denoiser_ddpm import GaussianDiffusion

class PnPDDPM:
    def __init__(self, config, model, diffusion, degradation, operator, noiser, device):
        self.config = config
        self.model = model
        self.diffusion = diffusion
        self.degradation = degradation
        self.operator = operator
        self.noiser = noiser
        self.device = device

    def __call__(self, g_x, y_n, record=False, save_root=None):
        samples = []

        iters_count_as_sample = np.linspace(
            self.config.num_burn_in_iters,
            self.config.num_iters-1,
            self.config.num_samples_per_run+1,
            dtype=int
        )[1:]

        assert self.config.num_iters - 1 in iters_count_as_sample, "num_iters-1 should be included in iters_count_as_sample"
        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar:
            rho_iter = self.config.rho * (self.config.rho_decay_rate ** i)
            rho_iter = max(rho_iter, self.config.rho_min)

            # likelihood step
            z = self.operator.proximal_generator(x, y_n, self.noiser.sigma, rho_iter) # not finish yet

            # prior step
            x = self.diffusion.p_sample_loop(
                model,
                y_n.shape,
                clip_denoised=False,
                model_kwargs={},
                orig_x=g_x,
                progress=True,
                degradation=self.degradation
                rho=rho_iter
            ).cpu()

            if i in iters_count_as_sample:
                samples.append(x)

        return samples[-1]