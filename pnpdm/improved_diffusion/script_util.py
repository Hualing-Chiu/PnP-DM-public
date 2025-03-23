import hydra.utils
from hydra.core.hydra_config import HydraConfig

from .samplers import denoiser_ddpm as gd

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    use_l1=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    input_sigma_t=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    # if use_kl:
    #     loss_type = gd.LossType.RESCALED_KL
    # elif rescale_learned_sigmas:
    #     loss_type = gd.LossType.RESCALED_MSE
    # elif use_l1:
    #     loss_type = gd.LossType.L1
    # else:
    #     loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        # loss_type=loss_type,
        input_sigma_t=input_sigma_t,
        rescale_timesteps=rescale_timesteps,
    )