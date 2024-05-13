#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import glob
import random
import logging
import math
import os
import torchvision
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm

import diffusers
from src.xray_decoder import AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers import AutoencoderKL
import open3d as o3d
from src.dataset import UpsamplerDataset

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    
    # The origin of all rays is the camera origin in world coordinate
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape) # (H, W, 3)

    return rays_o, rays_d

def depth_to_pcd_normals(GenDepths, GenNormals, GenColors):
    camera_angle_x = 0.8575560450553894
    image_width = GenDepths.shape[-1]
    image_height = GenDepths.shape[-2]
    fx = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
            2, image_height * image_width).T  # [h, w, 2]

    grid = rays_screen_coords.reshape(image_height, image_width, 2)

    cx = image_width / 2.0
    cy = image_height / 2.0

    i, j = grid[..., 1], grid[..., 0]

    directions = np.stack([(i-cx)/fx, -(j-cy)/fx, -np.ones_like(i)], -1) # (H, W, 3)

    c2w = np.eye(4).astype(np.float32)

    rays_origins, ray_directions = get_rays(directions, c2w)
    rays_origins = rays_origins[None].repeat(GenDepths.shape[0], 0)
    ray_directions = ray_directions[None].repeat(GenDepths.shape[0], 0)

    GenDepths = GenDepths.transpose(0, 2, 3, 1)
    GenNormals = GenNormals.transpose(0, 2, 3, 1)
    GenColors = GenColors.transpose(0, 2, 3, 1)
    
    valid_index = GenDepths[..., 0] > 0
    rays_origins = rays_origins[valid_index]
    ray_directions = ray_directions[valid_index]
    GenDepths = GenDepths[valid_index]
    normals = GenNormals[valid_index]
    colors = GenColors[valid_index]
    xyz = rays_origins + ray_directions * GenDepths

    return xyz, normals, colors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_model",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help=("the data root path."),
    )

    parser.add_argument(
        "--near",
        type=float,
        default=0.6,
        help=("the nearest distance"),
    )

    parser.add_argument(
        "--far",
        type=float,
        default=2.4,
        help=("the farest distance"),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    global_step = 0
    first_epoch = 0

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    vae_image = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    
    if args.pretrain_model is not None:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrain_model, subfolder="vae", revision=args.revision)
        global_step = int(args.pretrain_model.split("-")[-1])
        first_epoch = 0
    else:
        vae = AutoencoderKLTemporalDecoder.from_config("src/xray_decoder.json")
    
    vae.requires_grad_(True)
    vae_image.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    vae_image.to(accelerator.device, dtype=weight_dtype)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "vae"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = AutoencoderKLTemporalDecoder.from_pretrained(
                        input_dir, subfolder="vae", revision=args.revision)
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        vae.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = UpsamplerDataset(args.data_root, args.height, args.num_frames, near=args.near, far=args.far, phase="train")
    train_dataset[0]
    val_dataset = UpsamplerDataset(args.data_root, args.height, args.num_frames, near=args.near, far=args.far, phase="val")
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    vae, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        vae, optimizer, lr_scheduler, train_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    progress_bar.update(global_step)
    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(vae):
                xray_lr = batch["xray_lr"].to(weight_dtype).to(
                accelerator.device, non_blocking=True)

                xray_lr = xray_lr + torch.randn_like(xray_lr) * random.uniform(0, 1) * 0.1

                xray = batch["xray"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                conditional_pixel_values = batch["image_values"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True)

                # save xray_lr and conditional_pixel_values as images.
                if global_step % 100 == 0 and accelerator.is_main_process:
                    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
                    torchvision.utils.save_image(xray_lr[0, :, 0:1], os.path.join(args.output_dir, "samples", "depths_low.png"), normalize=True, nrow=4)
                    torchvision.utils.save_image(xray_lr[0, :, 1:4], os.path.join(args.output_dir, "samples", "colors_low.png"), normalize=True, nrow=4)
                    torchvision.utils.save_image(xray[0, :, 0:1], os.path.join(args.output_dir, "samples", "depths_high.png"), normalize=True, nrow=4)
                    torchvision.utils.save_image(xray[0, :, 1:4], os.path.join(args.output_dir, "samples", "normals_high.png"), normalize=True, nrow=4)
                    torchvision.utils.save_image(xray[0, :, 4:7], os.path.join(args.output_dir, "samples", "colors_high.png"), normalize=True, nrow=4)
                    torchvision.utils.save_image(conditional_pixel_values[0:1], os.path.join(args.output_dir, "samples", "images.png"), normalize=True)
                    # visual = torch.nn.functional.interpolate(xray_lr[0, :1, :1], (args.height, args.width))
                    # visual = visual.clip(-1, 1)
                    # visual = (visual + conditional_pixel_values[0:1]) / 2
                    # torchvision.utils.save_image(visual, os.path.join(args.output_dir, "samples", "pixel_value_alighed.png"), normalize=True)

                with torch.no_grad():
                    conditional_latents = vae_image.encode(conditional_pixel_values).latent_dist.mode()

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                conditional_latents = conditional_latents.unsqueeze(
                    1).repeat(1, xray_lr.shape[1], 1, 1, 1)
                xray_input = torch.cat(
                    [xray_lr, conditional_latents], dim=2)

                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
                    xray_input = xray_input.flatten(0, 1)
                    model_pred = vae(xray_input, num_frames=args.num_frames).sample
                    model_pred = model_pred.reshape(-1, args.num_frames, *model_pred.shape[1:])
                # MSE loss

                xray = xray.float()
                H = (xray[:, :, -1:] > 0.0).detach().expand(-1, -1, 7, -1, -1)
                hit_loss = F.binary_cross_entropy_with_logits(model_pred[:, :, -1:], xray[:, :, -1:] * 0.5 + 0.5)
                surface_loss = F.mse_loss(model_pred[:, :, :-1][H], xray[:, :, :-1][H])
                recon_loss = 0.1 * F.mse_loss(model_pred, xray)
                loss = hit_loss + surface_loss + recon_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.unwrap_model(vae).save_pretrained(save_path)
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # sample images!
                    if (
                        (global_step % args.validation_steps == 0)
                    ):
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        
                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        with torch.no_grad():
                            val_image_paths = val_dataset.depth_paths[:args.num_validation_images]
                            for val_img_idx in range(args.num_validation_images):
                                xray_lr = val_dataset[val_img_idx]["xray_lr"].to(accelerator.device, dtype=weight_dtype)[None]
                                xray_lr = xray_lr + torch.randn_like(xray_lr) * random.uniform(0, 1) * 0.1

                                xray_lr_0 = xray_lr[0]
                                GenDepths = (xray_lr_0[:, 0:1] * 0.5 + 0.5) * (args.far - args.near) + args.near
                                GenHits = ((GenDepths > args.near) * (GenDepths < args.far)).float()
                                GenDepths[GenHits == 0] = 0
                                GenDepths[GenDepths <= args.near] = 0
                                GenDepths[GenDepths >= args.far] = 0
                                GenNormals = F.normalize(xray_lr_0[:, 1:4], dim=1)
                                GenNormals[GenHits.repeat(1, 3, 1, 1) == 0] = 0
                                GenColors = (xray_lr_0[:, 1:4] * 0.5 + 0.5).clip(0, 1)
                                GenColors[GenHits.repeat(1, 3, 1, 1) == 0] = 0
                                GenDepths = GenDepths.cpu().numpy()
                                GenNormals = GenNormals.cpu().numpy()
                                GenColors = GenColors.cpu().numpy()
                                gen_pts, gen_normals, gen_colors = depth_to_pcd_normals(GenDepths, GenNormals, GenColors)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(gen_pts)
                                pcd.normals = o3d.utility.Vector3dVector(gen_normals)
                                pcd.colors = o3d.utility.Vector3dVector(gen_colors)
                                o3d.io.write_point_cloud(f"{val_save_dir}/step_{global_step}_val_img_{val_img_idx}_input.ply", pcd)

                                image_path = val_image_paths[val_img_idx].replace("depths", "images").replace(".npz", ".png")
                                image_val = load_image(image_path).convert("RGB").resize((args.width * 2, args.height * 2), Image.BILINEAR)
                                image_val.save(f"{val_save_dir}/step_{global_step}_val_img_{val_img_idx}_original.png")
                                
                                conditional_pixel_values = (torchvision.transforms.ToTensor()(image_val).unsqueeze(0) * 2 - 1).to(accelerator.device, dtype=weight_dtype)
                                conditional_latents = vae_image.encode(conditional_pixel_values).latent_dist.mode()

                                # Concatenate the `conditional_latents` with the `noisy_latents`.
                                conditional_latents = conditional_latents.unsqueeze(
                                    1).repeat(1, xray_lr.shape[1], 1, 1, 1)
                                xray_input = torch.cat(
                                    [xray_lr, conditional_latents], dim=2)
                                
                                xray_input = xray_input.flatten(0, 1)
                                model_pred = vae(xray_input, num_frames=args.num_frames).sample
                                outputs = model_pred.reshape(-1, args.num_frames, *model_pred.shape[1:])[0]

                                # save the generated images
                                outputs = outputs.clip(-1, 1)
                                GenDepths = (outputs[:, 0:1] * 0.5 + 0.5) * (args.far - args.near) + args.near
                                GenHits = (outputs[:, 7:8] > 0).float()
                                GenDepths[GenHits == 0] = 0
                                GenDepths[GenDepths <= args.near] = 0
                                GenDepths[GenDepths >= args.far] = 0
                                GenNormals = F.normalize(outputs[:, 1:4], dim=1)
                                GenNormals[GenHits.repeat(1, 3, 1, 1) == 0] = 0
                                GenColors = outputs[:, 4:7] * 0.5 + 0.5
                                GenColors[GenHits.repeat(1, 3, 1, 1) == 0] = 0
                                # torchvision.utils.save_image(GenHits, f"{val_save_dir}/step_{global_step}_val_img_{val_img_idx}_hits.png", normalize=True, nrow=4)
                                # torchvision.utils.save_image(GenDepths, f"{val_save_dir}/step_{global_step}_val_img_{val_img_idx}_depths.png", normalize=True, nrow=4)
                                # torchvision.utils.save_image(GenNormals, f"{val_save_dir}/step_{global_step}_val_img_{val_img_idx}_normals.png", normalize=True, nrow=4)
                                # torchvision.utils.save_image(GenColors, f"{val_save_dir}/step_{global_step}_val_img_{val_img_idx}_colors.png", normalize=True, nrow=4)
                                GenDepths = GenDepths.cpu().numpy()
                                GenNormals = GenNormals.cpu().numpy()
                                GenColors = GenColors.cpu().numpy()
                                gen_pts, gen_normals, gen_colors = depth_to_pcd_normals(GenDepths, GenNormals, GenColors)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(gen_pts)
                                pcd.normals = o3d.utility.Vector3dVector(gen_normals)
                                pcd.colors = o3d.utility.Vector3dVector(gen_colors)
                                o3d.io.write_point_cloud(f"{val_save_dir}/step_{global_step}_val_img_{val_img_idx}_prd.ply", pcd)

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            global_step += 1
            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()