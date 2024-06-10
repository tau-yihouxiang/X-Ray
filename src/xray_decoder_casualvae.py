# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalVAEMixin
from diffusers.utils import is_torch_version
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_3d_blocks import MidBlockTemporalDecoder, UpBlockTemporalDecoder
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution, Encoder

from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Decoder
from opensora.models.ae.videobase.utils.module_utils import resolve_str_to_obj, Module


class AutoencoderKLTemporalDecoder(ModelMixin, ConfigMixin, FromOriginalVAEMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        layers_per_block: (`int`, *optional*, defaults to 1): Number of layers per block.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        lr: float = 1e-5,
        hidden_size: int = 128,
        z_channels: int = 4,
        out_channels: int = 8,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = [],
        dropout: float = 0.0,
        resolution: int = 256,
        double_z: bool = True,
        embed_dim: int = 4,
        num_res_blocks: int = 2,
        loss_type: str = "opensora.models.ae.videobase.losses.LPIPSWithDiscriminator",
        loss_params: dict = {
            "kl_weight": 0.000001,
            "logvar_init": 0.0,
            "disc_start": 2001,
            "disc_weight": 0.5,
        },
        q_conv: str = "CausalConv3d",
        encoder_conv_in: Module = "CausalConv3d",
        encoder_conv_out: Module = "CausalConv3d",
        encoder_attention: Module = "AttnBlock3D",
        encoder_resnet_blocks: Tuple[Module] = (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        encoder_spatial_downsample: Tuple[Module] = (
            "SpatialDownsample2x",
            "SpatialDownsample2x",
            "SpatialDownsample2x",
            "",
        ),
        encoder_temporal_downsample: Tuple[Module] = (
            "",
            "TimeDownsample2x",
            "TimeDownsample2x",
            "",
        ),
        encoder_mid_resnet: Module = "ResnetBlock3D",
        decoder_conv_in: Module = "CausalConv3d",
        decoder_conv_out: Module = "CausalConv3d",
        decoder_attention: Module = "AttnBlock3D",
        decoder_resnet_blocks: Tuple[Module] = (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        decoder_spatial_upsample: Tuple[Module] = (
            "",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
        ),
        decoder_temporal_upsample: Tuple[Module] = ("", "", "TimeUpsample2x", "TimeUpsample2x"),
        decoder_mid_resnet: Module = "ResnetBlock3D",
    ):
        super().__init__()

        # pass init params to Decoder
        self.decoder = Decoder(
            z_channels=z_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=decoder_conv_in,
            conv_out=decoder_conv_out,
            attention=decoder_attention,
            resnet_blocks=decoder_resnet_blocks,
            spatial_upsample=decoder_spatial_upsample,
            temporal_upsample=decoder_temporal_upsample,
            mid_resnet=decoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        return_dict: bool = True,
        num_frames: int = 1,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        sample = sample.reshape(-1, num_frames, *sample.shape[1:])
        # permute
        sample = sample.permute(0, 2, 1, 3, 4)
        dec = self.decoder(sample)

        # permute
        dec = dec.permute(0, 2, 1, 3, 4)
        dec = dec.reshape(-1, *dec.shape[2:])

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
