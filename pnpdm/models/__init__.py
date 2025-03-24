from .edm.edm import create_edm_from_unet_adm
from .diffwave.diffwave import DiffWave

def get_model(name: str, **kwargs):
    if name == 'edm_from_unet_adm':
        return create_edm_from_unet_adm(**kwargs)
    elif name == 'diffwave':
        return DiffWave(**kwargs)
    else:
        raise NameError(f"Model {name} is not defined.")
