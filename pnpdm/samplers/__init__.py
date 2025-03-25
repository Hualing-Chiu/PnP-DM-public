# from .pnp_edm.pnp_edm import PnPEDM, PnPEDMBatch
# from .pnp_edm.pnp_edm_bh import PnPEDMBH, PnPEDMBHBatch
from .pnp_edm.pnp_ddpm import PnPDDPM

# def get_sampler(config, model, operator, noiser, device):
#     if config.name == 'pnp_edm':
#         return PnPEDM(config, model, operator, noiser, device)
#     elif config.name == 'pnp_edm_batch':
#         return PnPEDMBatch(config, model, operator, noiser, device)
#     elif config.name == 'pnp_edm_bh':
#         return PnPEDMBH(config, model, operator, noiser, device)
#     elif config.name == 'pnp_edm_batch_bh':
#         return PnPEDMBHBatch(config, model, operator, noiser, device)
#     else:
#         raise NameError(f"Model {config.name} is not defined.")

def get_sampler(config, model, diffusion, degradation, operator, noiser, device):
    if config.name == 'pnp_ddpm':
        return PnPDDPM(config, model, diffusion, degradation, operator, noiser, device)