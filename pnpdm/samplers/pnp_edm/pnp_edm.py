import torch, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from .denoiser_edm import Denoiser_EDM

import torchaudio

class PnPEDM:
    def __init__(self, config, model, operator, noiser, device):
        self.config = config
        self.model = model
        self.operator = operator
        self.noiser = noiser
        self.device = device
        if config.mode == 'vp':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.vp_kwargs, mode='pfode')
        elif config.mode == 've':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.ve_kwargs, mode='pfode')
        elif config.mode == 'iddpm':    
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='pfode')
        elif config.mode == 'edm':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.edm_kwargs, mode='pfode')
        elif config.mode == 'vp_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.vp_kwargs, mode='sde')
        elif config.mode == 've_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.ve_kwargs, mode='sde')
        elif config.mode == 'iddpm_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='sde')
        elif config.mode == 'edm_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.edm_kwargs, mode='sde')
        else:
            raise NotImplementedError(f"Mode {self.config.mode} is not implemented.")

    @property
    def display_name(self):
        return f'pnp-edm-{self.config.mode}-rho0={self.config.rho}-rhomin={self.config.rho_min}'

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):
        log = defaultdict(list)
        cmap = 'gray' if gt.shape[1] == 1 else None
        x = self.operator.initialize(gt, y_n)

        # logging
        x_save = inv_transform(x)
        z_save = torch.zeros_like(x_save)
        for name, metric in metrics.items():
            log[name].append(metric(x_save, inv_transform(gt)).item())
        
        xs_save = torch.cat((inv_transform(gt), x_save), dim=-1)
        try:
            zs_save = torch.cat((inv_transform(y_n.reshape(*gt.shape)), z_save), dim=-1)
        except:
            try:
                zs_save = torch.cat((inv_transform(self.operator.A_pinv(y_n).reshape(*gt.shape)), z_save), dim=-1)
            except:
                zs_save = torch.cat((z_save, z_save), dim=-1)

        if record:
            log["gt"] = inv_transform(gt).permute(0, 2, 3, 1).squeeze().cpu().numpy()
            log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        samples = []
        iters_count_as_sample = np.linspace(
            self.config.num_burn_in_iters, 
            self.config.num_iters-1, 
            self.config.num_samples_per_run+1, 
            dtype=int
        )[1:]
        assert self.config.num_iters-1 in iters_count_as_sample, "num_iters-1 should be included in iters_count_as_sample"
        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar: 
            rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
            rho_iter = max(rho_iter, self.config.rho_min)

            # likelihood step
            # proximal_generator return 的是 m_x + noise
            z = self.operator.proximal_generator(x, y_n, self.noiser.sigma, rho_iter)

            # prior step
            x = self.edm(z, rho_iter) # change

            if i in iters_count_as_sample:
                samples.append(x)

            # logging
            x_save = inv_transform(x)
            z_save = inv_transform(z)
            for name, metric in metrics.items():
                log[name].append(metric(x_save, inv_transform(gt)).item())
            sub_pbar.set_description(f'running PnP-EDM (xrange=[{x.min().item():.2f}, {x.max().item():.2f}], zrange=[{z.min().item():.2f}, {z.max().item():.2f}]) | psnr: {log["psnr"][-1]:.4f}')
            
            if i % (self.config.num_iters//10) == 0:
                xs_save = torch.cat((xs_save, x_save), dim=-1)
                zs_save = torch.cat((zs_save, z_save), dim=-1)
            
            if record:
                log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.plot(log["psnr"])
        plt.title(f'psnr (max): {np.amax(log["psnr"]):.4f}, (last): {log["psnr"][-1]:.4f}')
        plt.subplot(1, 3, 2)
        plt.plot(log["ssim"])
        plt.title(f'ssim (max): {np.amax(log["ssim"]):.4f}, (last): {log["ssim"][-1]:.4f}')
        plt.subplot(1, 3, 3)
        plt.plot(log["lpips"])
        plt.title(f'lpips (min): {np.amin(log["lpips"]):.4f}, (last): {log["lpips"][-1]:.4f}')
        plt.savefig(os.path.join(save_root, 'progress', fname+"_metrics.png"))
        plt.close()

        # logging
        xz_save = torch.cat((xs_save, zs_save), dim=-2).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        plt.imsave(os.path.join(save_root, 'progress', fname+"_x_and_z.png"), xz_save, cmap=cmap)
        np.save(os.path.join(save_root, 'progress', fname+"_log.npy"), log)

        return torch.concat(samples, dim=0)


class PnPEDMBatch(PnPEDM):
    @property
    def display_name(self):
        return f'pnp-edm-batch-{self.config.mode}-rho0={self.config.rho}-rhomin={self.config.rho_min}'

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):
        x = torch.randn(self.config.num_samples_per_run, *gt.shape[1:]).to(gt.device)

        nfe = []
        # logging
        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar:
            rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
            rho_iter = max(rho_iter, self.config.rho_min)

            # likelihood step
            z = self.operator.proximal_generator(x, y_n, self.noiser.sigma, rho_iter)

            # prior step
            x = self.edm(z, rho_iter)

        return x

class PnPDDPMSourceSeparation(PnPEDM):
    def __init__(self, config, model, operator, noiser, device):
        super().__init__(config, model, operator, noiser, device)

    def prepara_data(self, audio_files: List[str]):
        filtered_audio_files = [[file for file in files if file.endswith('wav')] for files in audio_files]
        n_samples = min([len(files) for files in filtered_audio_files])

        return {f"spk{i}": random.sample(files, k=n_samples)  for i, files in enumerate(filtered_mic2_audio_files)}

    def prepare_audio_before_degradation(self, x: List[torch.tensor]) -> torch.Tensor:
        min_sample_length = min(map(lambda tensor: tensor.size(-1), x))
        truncated_x = list(map(lambda tensor: tensor[..., :min_sample_length], x))
        return torch.cat(truncated_x, dim=0) # dim=-1 to dim=0

    def save_audios(
        self,
        pred_sample: torch.Tensor,
        degraded_sample: torch.Tensor,
        original_sample: torch.Tensor,
        idx: int,
        n_spk: int,
        sr: int = 16000,
    ):
        pred_chunked = torch.chunk(pred_sample, chunks=n_spk, dim=0)
        orig_chunked = torch.chunk(original_sample, chunks=n_spk, dim=0)
        for i, (cur_pred, cur_orig) in enumerate(zip(pred_chunked, orig_chunked)):
            name = f"Sample_{idx}_{i + 1}.wav"
            torchaudio.save(os.path.join(self.generated_path, name), cur_pred.view(1, -1), sr)
            torchaudio.save(os.path.join(self.original_path, name), cur_orig.view(1, -1), sr)

        name = f"Sample_{idx}.wav"
        torchaudio.save(os.path.join(self.degraded_path, name), degraded_sample.view(1, -1), sr)

    def degradation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([s for s in torch.chunk(x, 2, dim=0)]).sum(0)

    def __call__(
        self,
        audio_files: List[str],
        diffusion,
        target_sample_rate: int = 16000,
        segment_size: Optional[int] = None,
        device: str = "cpu"
    ):
        files_dict = self.prepare_data(audio_files)
        fake_samples = []
        real_samples = []

        sub_pbar = 
        for i, f in enumerate(zip(*files_dict.values())):
            x = self.load_audios(f, target_sample_rate, segment_size, device)
            x = self.prepare_audio_before_degradation(x)

            degraded_sample = self.degradation(x).cpu()

            # sample = diffusion.p_sample_loop(
            #     self.model,
            #     x.shape,
            #     clip_denoised=False,
            #     model_kwargs={},
            #     sample_method=self.task_type,
            #     orig_x=x,
            #     progress=True,
            #     degradation=self.degradation,
            # ).cpu()

            x = x.cpu()
            real_samples.append(x)
            fake_samples.append(sample)

            self.save_audios(sample, degraded_sample, x, i, len(audio_files), sr=target_sample_rate)
            del sample, x, degraded_sample
            torch.cuda.empty_cache()

        scores = calculate_all_metrics(fake_samples, self.metrics, reference_wavs=real_samples)
        log_results(results_dir=self.output_dir, res=scores)
