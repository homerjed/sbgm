from typing import Sequence
import equinox as eqx
from jaxtyping import Key
import ml_collections

from ._mixer import Mixer2d
from ._mlp import ResidualNetwork 
from ._unet import UNet
from ._unet_xy import UNetXY


def get_model(
    model_key: Key, 
    model_type: str, 
    data_shape: Sequence[int], 
    context_shape: Sequence[int], 
    parameter_dim: int,
    config: ml_collections.ConfigDict
) -> eqx.Module:
    if model_type == "Mixer":
        model = Mixer2d(
            data_shape,
            patch_size=config.model.patch_size,
            hidden_size=config.model.hidden_size,
            mix_patch_size=config.model.mix_patch_size,
            mix_hidden_size=config.model.mix_hidden_size,
            num_blocks=config.model.num_blocks,
            t1=config.t1,
            q_dim=context_shape,
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "UNet":
        model = UNet(
            data_shape=data_shape,
            is_biggan=config.model.is_biggan,
            dim_mults=config.model.dim_mults,
            hidden_size=config.model.hidden_size,
            heads=config.model.heads,
            dim_head=config.model.dim_head,
            dropout_rate=config.model.dropout_rate,
            num_res_blocks=config.model.num_res_blocks,
            attn_resolutions=config.model.attn_resolutions,
            final_activation=config.model.final_activation,
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "UNetXY":
        model = UNetXY(
            data_shape=data_shape,
            is_biggan=config.model.is_biggan,
            dim_mults=config.model.dim_mults,
            hidden_size=config.model.hidden_size,
            heads=config.model.heads,
            dim_head=config.model.dim_head,
            dropout_rate=config.model.dropout_rate,
            num_res_blocks=config.model.num_res_blocks,
            attn_resolutions=config.model.attn_resolutions,
            final_activation=config.model.final_activation,
            q_dim=context_shape[0], # Just grab channel assuming 'q' is a map like x
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "mlp":
        model = ResidualNetwork(
            in_size=np.prod(data_shape),
            width_size=config.model.width_size,
            depth=config.model.depth,
            activation=config.model.activation,
            dropout_p=config.model.dropout_p,
            y_dim=parameter_dim,
            key=model_key
        )
    if model_type == "DiT":
        raise NotImplementedError
    return model