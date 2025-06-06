import torch, os, hydra, logging
import torchaudio
import itertools
import random
import json
import gc
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from torchvision import transforms
from pnpdm.data import get_dataset, get_dataloader
from pnpdm.tasks import get_operator, get_noise, MotionBlurCircular
from pnpdm.models import get_model
from pnpdm.samplers import get_sampler
from hydra.core.hydra_config import HydraConfig
from monai.metrics import PSNRMetric, SSIMMetric
from taming.modules.losses.lpips import LPIPS
from pnpdm.improved_diffusion.inference_utils import calculate_all_metrics, log_results, remove_prefix_from_state_dict
from pnpdm.data.utils import cut_audio_segment
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
    metrics_list = [
        hydra.utils.instantiate(task_config.metrics[metric], device=device)
        for metric in task_config.metrics
    ]
    # print([type(m) for m in metrics_list])
    # source separation load data
    # if task_config.operator.name == "source_separation":
    #     audio_files = [list(map(lambda x: os.path.join(d, x), os.listdir(d))) for d in data_config.root] # List[str]

    # files_dict = prepara_data(audio_files) 
    if task_config.operator.name == "source_separation":
        audio_files = prepare_data(data_config.root[0])  # List[List[str]]

    model = get_model(model_config.name, **model_config.model)
    # load checkpoint
    pl_ckpt = torch.load(model_config.model_path, map_location="cpu")
    # model_state = remove_prefix_from_state_dict(
    #     pl_ckpt["state_dict"], j=1
    # )
    # load model
    model.load_state_dict(pl_ckpt)
    model = model.to(device)
    model.eval()

    # load ddpm
    diffusion = hydra.utils.call(cfg.diffusion) # GaussianDiffusion
    # load sampler
    sampler = get_sampler(sampler_config, model=model, diffusion=diffusion, degradation=degradation, operator=operator, noiser=noiser, device=device)

    # inference
    output_dir = os.path.join("results_libritts_new_model_2_spk_no_spk_condition", task_config.operator.name) # results_vctk_820k_finetune_grad_spk_condition
    generated_path = os.path.join(output_dir, "generated")
    original_path = os.path.join(output_dir, "original")
    degraded_path = os.path.join(output_dir, "degraded")
    for path in [generated_path, original_path, degraded_path]:
        if not exists(path):
            os.makedirs(path)
 
    audio_files = sorted(audio_files.items())
    batch_size = cfg.batch_size
    # inference
    generated_samples = []
    real_samples = []
    
    for batch_start in range(0, len(audio_files), batch_size):
        if batch_start + batch_size > len(audio_files):
            batch_size = len(audio_files) - batch_start

        batch_group = audio_files[batch_start:batch_start + batch_size]
        x_batch = [load_audios(f, 16000, None, "cpu") for _, f in batch_group]
        x_batch = [prepare_audio_before_degradation(x) for x in x_batch]
        # min_len = min(x.shape[-1] for x in x_batch)
        x_batch = torch.stack(x_batch).flatten(0, 1).to(device)
        print(f"x_batch shape: {x_batch.shape}")
        degraded_batch = degradation(x_batch).cpu()  # y_n

        sample_list = []
        for _ in tqdm(range(cfg.num_runs)):
            sample = sampler(
                g_x=x_batch,
                y_n=degraded_batch,
                record=cfg.record,
                save_root=generated_path,
                # task_kwargs={
                #     "ref": ref_batch,
                #     "mask_ref": mask_ref_batch,
                # }
                task_kwargs=None
            )

            sample_list.append(sample)
            del sample
            torch.cuda.empty_cache()
            gc.collect()

        n_spk = int(x_batch.shape[0] / batch_size)
        base_sample = sample_list[0]
        for b in range(len(batch_group)):
            base = base_sample[b * n_spk:(b + 1) * n_spk]
            summed = base.clone()
            for j in range(1, len(sample_list)):
                current = sample_list[j][b * n_spk:(b + 1) * n_spk]
                best_score = float('-inf')
                for perm in itertools.permutations(range(n_spk)):
                    reordered = current[list(perm)]
                    score = sum(sisnr(base[k], reordered[k]) for k in range(n_spk))
                    if score > best_score:
                        best_score = score
                        best_perm = perm
                current = current[list(best_perm)]
                summed += current
            avg = summed / cfg.num_runs
            generated_samples.append(avg.cpu())
            real_samples.append(x_batch[b * n_spk:(b + 1) * n_spk].cpu())  # speaker num

            save_audios(
                original_path,
                generated_path,
                degraded_path,
                # reference_path,
                avg,
                degraded_batch[b],
                x_batch[b * n_spk:(b + 1) * n_spk],
                # ref_batch[b * n_spk:(b + 1) * n_spk],
                batch_start + b,
                n_spk,
                16000
            )
        del sample_list, avg, summed, x_batch, degraded_batch
        torch.cuda.empty_cache()

    scores = calculate_all_metrics(
        generated_samples, metrics_list, reference_wavs=real_samples
    )
    log_results(results_dir=output_dir, res=scores)

def exists(path: str):
        return os.path.exists(path)

def prepare_data(audio_files: List[str]):
    pattern = re.compile(r"Sample_(\d+)_\d+\.wav")
    file_path = glob.glob(os.path.join(audio_files, "Sample_*_*.wav"))

    audio_dict = defaultdict(list)
    for path in file_path:
        match = pattern.search(path)
        if match:
            i = int(match.group(1))
            audio_dict[i].append(path)

    audio_dict = {k: sorted(v) for k, v in audio_dict.items() if len(v) == 2}  # filter out incomplete samples
    return audio_dict
# def prepara_data(audio_files: List[str]):
#     filtered_audio_files = [[file for file in files if "mic1" in file] for files in audio_files]
#     # filtered_audio_files = [[file for file in files if file.endswith('wav')] for files in audio_files]
#     n_samples = min([len(files) for files in filtered_audio_files])

#     return {f"spk{i}": random.sample(files, k=n_samples)  for i, files in enumerate(filtered_audio_files)}

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
    # x = torchaudio.functional.vad(x, target_sample_rate)
    x = x.to(device).unsqueeze(0)
    return x

def load_audios(paths: List[str], *args, **kwargs) -> List[torch.Tensor]:
    return [load_audio(p, *args, **kwargs) for p in paths]

def save_audios(
    original_path: str,
    generated_path: str,
    degraded_path: str,
    pred_sample: torch.Tensor,
    degraded_sample: torch.Tensor,
    original_sample: torch.Tensor,
    idx: int,
    n_spk: int,
    sr: int,
    # train_mean: float,
    # train_std: float,
    # input_mean: float,
    # input_std: float,
):
    pred_chunked = torch.chunk(
        pred_sample, chunks=n_spk, dim=0 # dim=0 -> batch # modify
    )  # explicit number of chunks 2
    orig_chunked = torch.chunk(
        original_sample, chunks=n_spk, dim=0
    )  # explicit number of chunks 2

    for i, (cur_pred, cur_orig) in enumerate(zip(pred_chunked, orig_chunked)):
        # denormalize
        # cur_pred = (cur_pred - train_mean) / (train_std + 1e-9)
        # cur_pred = cur_pred * input_std[i].to('cpu') + input_mean[i].to('cpu')

        name = f"Sample_{idx}_{i + 1}.wav"
        torchaudio.save(
            os.path.join(generated_path, name), cur_pred.detach().cpu().view(1, -1), sr
        )
        print(os.path.join(generated_path, name))
        torchaudio.save(
            os.path.join(original_path, name), cur_orig.detach().cpu().view(1, -1), sr
        )

    # redefine name for degraded
    name = f"Sample_{idx}.wav"
    torchaudio.save(
        os.path.join(degraded_path, name), degraded_sample.detach().cpu().view(1, -1), sr
    )

def degradation(x: torch.Tensor) -> torch.Tensor:
    n_spk = 2
    B = x.shape[0] // n_spk
    x_grouped = x.view(B, n_spk, *x.shape[1:])  # [B, n_spk, C, T]
    return x_grouped.sum(dim=1)  # sum over speakers
    # return torch.stack([s for s in torch.chunk(x, 2, dim=0)]).sum(0)

def sisnr(x, y):
        alpha = (x * y).sum(-1, keepdims=True) / (
            x.square().sum(-1, keepdims=True) + 1e-9
        )
        real_samples_scaled = alpha * x
        e_target = real_samples_scaled.square().sum(-1)
        e_res = (real_samples_scaled - y).square().sum(-1)
        return 10 * torch.log10(e_target / (e_res + 1e-9)).cpu().numpy()

if __name__ == '__main__':
    try:
        posterior_sample()
    except Exception as e:
        print(f"Error: {e}")