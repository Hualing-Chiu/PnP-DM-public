import torch, os, hydra, logging, json
import torchaudio
import itertools
import random
import gc
import re, glob
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
    # source separation load data
    if task_config.operator.name == "source_separation":
        # audio_files = [list(map(lambda x: os.path.join(d, x), os.listdir(d))) for d in data_config.root] # List[str]
        audio_files = prepare_data(data_config.root[0]) # List[str] or str
        # print(f"Audio files: {audio_files}")
        ref_files = prepare_data(data_config.root[1]) # List[str] or str
    
    # print(f"Audio files: {audio_files}")
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
    output_dir = os.path.join(cfg.output_dir, task_config.operator.name) # results_vctk_new_model_2spk_no_condition_no_loss_10
    generated_path = os.path.join(output_dir, "generated")
    original_path = os.path.join(output_dir, "original")
    degraded_path = os.path.join(output_dir, "degraded")

    for path in [generated_path, original_path, degraded_path]:
        if not exists(path):
            os.makedirs(path)
 
    # audio_files = sorted(audio_files.items())
    # inference
    num_runs = cfg.num_runs
    generated_samples = []
    real_samples = []
    # files_key = list(files_dict.keys()) # [spk0, spk1, ...]

    for (i, f), (j, r) in zip(sorted(audio_files.items()), sorted(ref_files.items())):
        # print(len(f), len())
        assert i == j, f"Index mismatch: {i} != {j}"
        x = load_audios(f, 16000, None, "cpu", False)
        print(f"X_shape: {x[0].shape}, {x[1].shape}")
        x = prepare_audio_before_degradation(x)
        print(f"X_shape: {x[0].shape}, {x[1].shape}")
        degraded_sample = degradation(x, task_config.operator.n_spk).cpu() # y_n

        ref = load_audios(r, 16000, None, "cpu", True)
        ref = prepare_audio_before_degradation(ref) # [2 ,1 ,T]
        mask_ref = torch.zeros_like(ref).to(torch.bool)

        # repeat for batch inference and stack outputs
        # x_repeat = x.repeat(num_runs, 1, 1)  # [num_runs * n_spk, 1, T]
        # degraded_repeat = degraded_sample.repeat(num_runs, 1, 1)  # [num_runs * n_spk, 1, T]
        # ref_repeat = ref.repeat(num_runs, 1, 1).squeeze(1)  # [num_runs * n_spk, 1, T]
        # mask_ref_repeat = mask_ref.repeat(num_runs, 1, 1).squeeze(1)  # [num_runs * n_spk, T]
        # print(x_repeat.shape, degraded_repeat.shape)
        # print(ref_repeat.shape, mask_ref_repeat.shape)
        # 分次處理避免 OOM
        partial_runs = 5
        num_chunks = num_runs // partial_runs

        sample_chunks = []
        for _ in range(num_chunks):
            x_repeat = x.repeat(partial_runs, 1, 1)
            degraded_repeat = degraded_sample.repeat(partial_runs, 1, 1)
            ref_repeat = ref.repeat(partial_runs, 1, 1).squeeze(1)
            mask_ref_repeat = mask_ref.repeat(partial_runs, 1, 1).squeeze(1)

            samples = sampler(
                g_x=x_repeat.to(device),
                y_n=degraded_repeat.to(device),
                record=cfg.record,
                save_root=generated_path,
                # task_kwargs={
                #     "ref": ref_repeat.to(device),
                #     "mask_ref": mask_ref_repeat.to(device),
                # }
                task_kwargs=None
            )  # [partial_runs * n_spk, 1, T]
            sample_chunks.append(samples.cpu())
            gc.collect()
            torch.cuda.empty_cache()

        # 拼接所有 samples
        samples = torch.cat(sample_chunks, dim=0)  # [num_runs * n_spk, 1, T]
        # sampling
        # sample_list = []
        # samples = sampler(
        #     g_x=x_repeat.to(device),
        #     y_n=degraded_repeat.to(device),
        #     record=cfg.record,
        #     save_root=generated_path,
        #     task_kwargs= {
        #         "ref": ref_repeat.to(device),
        #         "mask_ref": mask_ref_repeat.to(device),
        #     }
        #     # task_kwargs=None
        # ) # [num_runs * n_spk, 1, T]

        x = x.cpu()
        real_samples.append(x)
        n_spk = x.shape[0] # speaker num

        # permutation alignment and average
        samples = samples.view(num_runs, n_spk, 1, -1)  # [num_runs, n_spk, 1, T]
        base_sample = samples[0]
        summed_sample = base_sample.clone()
        for r in range(1, num_runs):
            sample_next = samples[r]
            best_score = float('-inf')
            best_perm = None
            for perm in itertools.permutations(range(n_spk)):
                reordered = sample_next[list(perm)]
                score = sum(sisnr(base_sample[k], reordered[k]) for k in range(n_spk))
                if score > best_score:
                    best_score = score
                    best_perm = perm
            sample_next = sample_next[list(best_perm)]
            summed_sample += sample_next

        samples_mean = summed_sample / num_runs
        generated_samples.append(samples_mean)
        save_audios(
            original_path,
            generated_path,
            degraded_path,
            samples_mean,
            degraded_sample,
            x,
            ref,
            i,
            n_spk,
            sr=16000,
        )
        # torch.cuda.empty_cache()
        del sample_chunks, samples_mean, summed_sample, x, degraded_sample, base_sample, samples, sample_next, x_repeat, degraded_repeat
        del ref, mask_ref, ref_repeat, mask_ref_repeat
        gc.collect()
        torch.cuda.empty_cache()

    scores = calculate_all_metrics(
        generated_samples, metrics_list, reference_wavs=real_samples
    )
    log_results(results_dir=output_dir, res=scores)

def exists(path: str):
        return os.path.exists(path)

# def prepare_data(audio_files: List[str]):
#     filtered_audio_files = [[file for file in files if "mic2" in file] for files in audio_files]
#     # filtered_audio_files = [[file for file in files if file.endswith('wav')] for files in audio_files]
#     n_samples = min([len(files) for files in filtered_audio_files])

#     return {f"spk{i}": random.sample(files, k=n_samples)  for i, files in enumerate(filtered_audio_files)}

def prepare_data(audio_files: List[str]):
    pattern = re.compile(r"Sample_(\d+)_\d+\.wav")
    file_path = glob.glob(os.path.join(audio_files, "Sample_*_*.wav"))
    file_path = sorted(file_path, key=lambda x: int(x.split('/')[-1].split('_')[1]))  # sort by Sample index
    audio_dict = defaultdict(list)
    for path in file_path:
        # print(f'current path: {path}')
        # print(path.split('_')[1])
        # if int(path.split('/')[-1].split('_')[1]) > 586: # 2 spk: 965、 3 spk: 596
        match = pattern.search(path)    
        if match:
            i = int(match.group(1))
            audio_dict[i].append(path)

    audio_dict = {k: sorted(v) for k, v in audio_dict.items() if len(v) == 2}  # filter out incomplete samples
    # print(audio_dict)
    return audio_dict # {0: ['.../Sample_0_1.wav', '.../Sample_0_2.wav'], 1: []...}

def prepare_audio_before_degradation(x: List[torch.Tensor]) -> torch.Tensor:
    # max_length = 150000  # 12 seconds at 16 kHz
    # truncated_x = []

    # for tensor in x:
    #     if tensor.size(-1) > max_length:
    #         truncated_tensor = tensor[..., :max_length]
    #     else:
    #         truncated_tensor = tensor
    #     truncated_x.append(truncated_tensor)

    # min_sample_length = min(t.size(-1) for t in truncated_x)
    # truncated_x = [t[..., :min_sample_length] for t in truncated_x]

    # return torch.cat(truncated_x, dim=0)
    min_sample_length = min(map(lambda tensor: tensor.size(-1), x))
    truncated_x = list(map(lambda tensor: tensor[..., :min_sample_length], x))

    return torch.cat(truncated_x, dim=0) # dim=-1 to dim=0

def load_audio(
    path: str,
    target_sample_rate: int = 16000,
    segment_size: Optional[int] = None,
    device: str = 'cpu',
    is_ref: bool = False
) -> torch.Tensor:
    print(path)
    x, sr = torchaudio.load(path)
    x = torchaudio.functional.resample(x, sr, target_sample_rate)
    if segment_size is not None:
        x = cut_audio_segment(x, segment_size)
    # x = torchaudio.functional.vad(x, target_sample_rate)
    if is_ref:
        x = (x - x.mean()) / x.std() * 1
    else:
        x = (x - x.mean()) / x.std() * 0.2
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
    ref: torch.Tensor,
    idx: int,
    n_spk: int,
    sr: int,
):
    pred_chunked = torch.chunk(
        pred_sample, chunks=n_spk, dim=0 # dim=0 -> batch # modify
    )  # explicit number of chunks 2
    orig_chunked = torch.chunk(
        original_sample, chunks=n_spk, dim=0
    )  # explicit number of chunks 2
    # ref_chunked = torch.chunk(
    #     ref, chunks=n_spk, dim=0
    # )  # explicit number of chunks 2

    for i, (cur_pred, cur_orig) in enumerate(zip(pred_chunked, orig_chunked)):

        name = f"Sample_{idx}_{i + 1}.wav"
        torchaudio.save(
            os.path.join(generated_path, name), cur_pred.detach().cpu().view(1, -1), sr
        )
        torchaudio.save(
            os.path.join(original_path, name), cur_orig.detach().cpu().view(1, -1), sr
        )
    
    # redefine name for degraded
    name = f"Sample_{idx}.wav"
    torchaudio.save(
        os.path.join(degraded_path, name), degraded_sample.detach().cpu().view(1, -1), sr
    )

def degradation(x: torch.Tensor, n_spk: int) -> torch.Tensor:
    return torch.stack([s for s in torch.chunk(x, n_spk, dim=0)]).sum(0)

def sisnr(x, y):
        alpha = (x * y).sum(-1, keepdims=True) / (
            x.square().sum(-1, keepdims=True) + 1e-9
        )
        real_samples_scaled = alpha * x
        e_target = real_samples_scaled.square().sum(-1)
        e_res = (real_samples_scaled - y).square().sum(-1)
        return 10 * torch.log10(e_target / (e_res + 1e-9)).detach().cpu().numpy()

if __name__ == '__main__':
    try:
        posterior_sample()
    except Exception as e:
        print(f"Error: {e}")