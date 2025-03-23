import torch, os, hydra, logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from torchvision import transforms
from pnpdm.data import get_dataset, get_dataloader
from pnpdm.tasks import get_operator, get_noise, get_metrics, MotionBlurCircular
from pnpdm.models import get_model
from pnpdm.samplers import get_sampler
from hydra.core.hydra_config import HydraConfig
from monai.metrics import PSNRMetric, SSIMMetric
from taming.modules.losses.lpips import LPIPS
from pnpdm.improved_diffusion.inference_utils import calculate_all_metrics, log_results
from pnpdm.improved_diffusion.metrics import Metric

@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def posterior_sample(cfg):
    # load configurations
    data_config = cfg.data
    task_config = cfg.task
    model_config = cfg.model
    sampler_config = cfg.sampler
    # device setting
    device_str = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    # prepare task (forward model and noise)
    operator = get_operator(**task_config.operator, device=device)
    noiser = get_noise(**task_config.noise)
    metrics = get_metrics(**task_config.metrics)
    print(f"metrics: {metrics}")

    # prepare dataloader
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.Normalize((0.5), (0.5))
    # ])
    # inv_transform = transforms.Compose([
    #     transforms.Normalize((-1), (2)),
    #     transforms.Lambda(lambda x: x.clamp(0, 1).detach())
    # ])
    
    # dataset = get_dataset(**data_config, transform=transform)
    # num_test_images = len(dataset)
    # dataloader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # source separation load data
    if task_config.operator.name == "source_separation":
        audio_files = [list(map(lambda x: os.path.join(d, x), os.listdir(d))) for d in data_config.root] # List[str]

    files_dict = prepara_data(audio_files)
    # load model
    model = get_model(**model_config)
    model = model.to(device)
    model.eval()

    # load ddpm
    diffusion = hydra.utils.call(cfg.diffusion) # GaussianDiffusion
    # load sampler
    sampler = get_sampler(sampler_config, model=model, diffusion=diffusion, degradation=degradation, operator=operator, noiser=noiser, device=device)

    # inference
    output_dir = os.path.join("results", task_config.operator.name)
    generated_path = os.path.join(output_dir, "generated")
    original_path = os.path.join(output_dir, "results")
    degraded_path = os.path.join(output_dir, "degraded")
    for path in [generated_path, original_path, degraded_path]:
        if not exists(path):
            os.makedirs(path)

    # inference
    fake_samples = []
    real_samples = []
    files_key = list(files_dict.keys())

    for i, f in enumerate(zip(*files_dict.values())):
        x = load_audios(f, 16000, None, "cpu")
        x = prepare_audio_before_degradation(x)
        degraded_sample = degradation(x).cpu() # y_n

        # sampling
        for _ in tqdm(range(cfg.num_runs)):
            sample = sampler(
                g_x=x,
                y_n=degraded_sample,
                record=cfg.record,
                save_root=generated_path
            )

        x = x.cpu()
        real_samples.append(x)
        generated_samples.append(sample)

        save_audios(sample, degraded_sample, x, i, len(audio_files), sr=16000)

        del sample, x, degraded_sample
        torch.cuda.empty_cache()

    scores = calculate_all_metrics(
        generated_samples, List[Metric], reference_wavs=real_samples
    )
    log_results(results_dir=output_dir, res=scores)

def exists(path: str):
        return os.path.exists(path)

def prepara_data(audio_files: List[str]):
    filtered_mic2_audio_files = [[file for file in files if "mic1" in file] for files in audio_files]
    # filtered_audio_files = [[file for file in files if file.endswith('wav')] for files in audio_files]
    n_samples = min([len(files) for files in filtered_mic2_audio_files])

    return {f"spk{i}": random.sample(files, k=n_samples)  for i, files in enumerate(filtered_mic2_audio_files)}

def prepare_audio_before_degradation(x: List[torch.Tensor]) -> torch.Tensor:
    min_sample_length = min(map(lambda tensor: tensor.size(-1), x))
    truncated_x = list(map(lambda tensor: tensor[..., :min_sample_length], x))
    return torch.cat(truncated_x, dim=0) # dim=-1 to dim=0

def load_audio(
    path: str,
    target_sample_rate: int = 16000,
    segment_size: Optional[int] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    print(path)
    x, sr = torchaudio.load(path)
    x = torchaudio.functional.resample(x, sr, target_sample_rate)
    if segment_size is not None:
        x = cut_audio_segment(x, segment_size)
    x = torchaudio.functional.vad(x, target_sample_rate)
    x = x.to(device).unsqueeze(0)
    return x

def load_audios(paths: List[str], *args, **kwargs) -> List[torch.Tensor]:
    return [self.load_audio(p, *args, **kwargs) for p in paths]

def save_audios(
    self,
    pred_sample: torch.Tensor,
    degraded_sample: torch.Tensor,
    original_sample: torch.Tensor,
    idx: int,
    n_spk: int,
    sr: int = 16000,
):
    pred_chunked = torch.chunk(
        pred_sample, chunks=n_spk, dim=0 # dim=0 -> batch # modify
    )  # explicit number of chunks 2
    orig_chunked = torch.chunk(
        original_sample, chunks=n_spk, dim=0
    )  # explicit number of chunks 2
    for i, (cur_pred, cur_orig) in enumerate(zip(pred_chunked, orig_chunked)):
        name = f"Sample_{idx}_{i + 1}.wav"
        torchaudio.save(
            os.path.join(self.generated_path, name), cur_pred.view(1, -1), sr
        )
        torchaudio.save(
            os.path.join(self.original_path, name), cur_orig.view(1, -1), sr
        )
    
    # concate the separate audio
    concatenated_pred = torch.cat(pred_chunked, dim=-1)
    name = f"Sample_{idx}.wav"
    torchaudio.save(
        os.path.join(self.concatenate_path, name), concatenated_pred.view(1, -1), sr
    )

    # redefine name for degraded
    name = f"Sample_{idx}.wav"
    torchaudio.save(
        os.path.join(self.degraded_path, name), degraded_sample.view(1, -1), sr
    )

def degradation(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([s for s in torch.chunk(x, 2, dim=0)]).sum(0)

if __name__ == '__main__':
    try:
        posterior_sample()
    except Exception as e:
        print(f"Error: {e}")