from .quijote import QuijoteConfig
from .mnist import MNISTConfig
from .cifar10 import CIFAR10Config 
from .flowers import FlowersConfig 
from .moons import MoonsConfig
from .dgdm import DgdmConfig 
from .grfs import GRFConfig

Config = QuijoteConfig | MoonsConfig | MNISTConfig | CIFAR10Config | FlowersConfig | GRFConfig

__all__ = [
    Config, 
    QuijoteConfig, 
    MoonsConfig, 
    MNISTConfig, 
    CIFAR10Config,
    FlowersConfig,
    DgdmConfig,
    GRFConfig
]