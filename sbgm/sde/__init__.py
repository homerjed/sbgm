from ml_collections import ConfigDict

from . import _sde
from ._sde import SDE
from ._ve import VESDE
from ._vp import VPSDE
from ._subvp import SubVPSDE


def get_sde(config: ConfigDict) -> _sde.SDE:
    assert config.sde in ["VE", "VP", "SubVP"]
    name = config.sde + "SDE"
    sdes = [VESDE, VPSDE, SubVPSDE]
    sde = sdes[[sde.__name__ for sde in sdes].index(name)]
    return sde(
        beta_integral_fn=config.beta_integral,
        dt=config.dt,
        t0=config.t0, 
        t1=config.t1
    )