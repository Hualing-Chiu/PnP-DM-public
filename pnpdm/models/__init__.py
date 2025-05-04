from .edm.edm import create_edm_from_unet_adm
from .diffwave.diffwave import DiffWave
from .unet_libritts.unet_libritts import UNetModel

def get_model(name: str, **kwargs):
    if name == 'edm_from_unet_adm':
        return create_edm_from_unet_adm(**kwargs)
    elif name == 'diffwave':
        return DiffWave(**kwargs)
    elif name == 'unet_libritts':
        return UNetModel(**kwargs)
    else:
        raise NameError(f"Model {name} is not defined.")
