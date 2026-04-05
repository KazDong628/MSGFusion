"""
Train MSGFusion on paired infrared/visible patches with CLIP-derived embeddings.

Loss weights and data flow match the original release; identifiers follow a
paper-style layout under the ``msgfusion`` package.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import trange
import pytorch_msssim
import scipy.io as scio
from PIL import Image
import torchvision.models as models

import third_party_clip.clip as clip_backend

from msgfusion.config_shim import runtime_config as cfg
from msgfusion.data.preprocessing import (
    enumerate_training_sample_ids,
    load_grayscale_training_tensor,
    shuffle_trim_to_batches,
)
from msgfusion.models.dense_fuse import DenseFuseNet
from msgfusion.models.fusion_network import build_msgfusion_network
from msgfusion.visualization.plot_training_loss import showLossChart

_KSIZE_LOCAL_STAT = 9
_PAD_LOCAL_STAT = 4

_W_IR = 2.2
_W_VI = 1.2
_W_CONTRAST = 0.3
_DENSE_PRIOR_EPS = 1e-8


def load_pretrained_densefuse_encoder(ckpt_file: str, device: str) -> DenseFuseNet:
    """Load DenseFuse for spatial soft weights (frozen)."""
    net = DenseFuseNet()
    net.load_state_dict(torch.load(ckpt_file, map_location=device))
    net.eval()
    return net.to(device)


def local_contrast_map(feature: torch.Tensor) -> torch.Tensor:
    mean_map = F.avg_pool2d(feature, _KSIZE_LOCAL_STAT, 1, _PAD_LOCAL_STAT)
    mean_sq = F.avg_pool2d(feature * feature, _KSIZE_LOCAL_STAT, 1, _PAD_LOCAL_STAT)
    variance = mean_sq - mean_map * mean_map
    return torch.sqrt(torch.clamp(variance, min=1e-6))


def read_binary_roi_mask(path_png: str) -> torch.Tensor:
    pil = Image.open(path_png).convert("L")
    arr = np.asarray(pil, dtype=np.float32)
    arr = (arr > 127).astype(np.float32)
    return torch.from_numpy(arr)[None, None]


def run_msgfusion_training_loop() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.save_model_dir, exist_ok=True)
    os.makedirs(cfg.save_loss_dir, exist_ok=True)

    _clip_encoder, _ = clip_backend.load("ViT-B/32", device=compute_device)

    caption_bank = torch.load(
        "./embeddings/caption_train/caption_embeddings_train_LLVIP.pt"
    ).to(compute_device)
    roi_embedding_bank = torch.load(
        "./embeddings/visual_train/visual_embedding_train_LLVIP.pt"
    ).to(compute_device)

    student = build_msgfusion_network().to(compute_device)
    solver = Adam(student.parameters(), lr=cfg.lr)

    warmup_restarts = CosineAnnealingWarmRestarts(
        solver,
        T_0=2000,
        T_mult=2,
        eta_min=0.0,
    )

    global_step = 0
    pixel_mse = torch.nn.MSELoss(reduction="mean")
    ssim_loss = pytorch_msssim.MS_SSIM()

    teacher_densefuse = load_pretrained_densefuse_encoder("./models/DenseFuse.model", compute_device)

    vgg19_features = models.vgg19(pretrained=True).features.to(compute_device).eval()
    vgg19_shallow = vgg19_features[:9]
    for p in vgg19_shallow.parameters():
        p.requires_grad_(False)

    train_ir_vis_root = "./IVT_train_LLVIP"
    assoc_mask_root = "./IVT_train_association/association"
    ordered_ids = enumerate_training_sample_ids()
    minibatch = cfg.batch_size

    running_loss_log = []
    captions_per_sample = 5

    epoch_bar = trange(cfg.epochs)
    for epoch in epoch_bar:
        student.train()
        flat_indices, n_minibatches = shuffle_trim_to_batches(ordered_ids, minibatch)

        acc_loss_window = 0.0
        step_in_epoch = 0

        for mb in range(n_minibatches):
            slice_ids = flat_indices[mb * minibatch : (mb + 1) * minibatch]

            for slot in range(minibatch):
                pid = int(slice_ids[slot])
                path_ir = os.path.join(train_ir_vis_root, "ir", f"{pid}.png")
                path_vis = os.path.join(train_ir_vis_root, "vis", f"{pid}.png")

                cube_ir = F.interpolate(
                    load_grayscale_training_tensor(path_ir), scale_factor=0.5, mode="bilinear"
                ).to(compute_device)
                cube_vis = F.interpolate(
                    load_grayscale_training_tensor(path_vis), scale_factor=0.5, mode="bilinear"
                ).to(compute_device)

                assoc_dir = f"IVT_LLVIP_2000_imageIndex_{pid}_textIndex_5"
                roi_map_path = os.path.join(
                    assoc_mask_root, assoc_dir, "Final_Finetuned_BinaryInterestedMap.png"
                )
                roi_gate = F.interpolate(
                    read_binary_roi_mask(roi_map_path),
                    scale_factor=0.5,
                    mode="nearest",
                ).to(compute_device)

                cap_off = (pid - 1) * captions_per_sample
                cap_slice = caption_bank[cap_off : cap_off + captions_per_sample]
                roi_slice = roi_embedding_bank[cap_off : cap_off + captions_per_sample]

                solver.zero_grad()

                with torch.no_grad():
                    enc_ir = teacher_densefuse.encoder(cube_ir)[0]
                    enc_vis = teacher_densefuse.encoder(cube_vis)[0]
                    score_ir = enc_ir.sum(dim=1, keepdim=True)
                    score_vis = enc_vis.sum(dim=1, keepdim=True)
                    z = torch.exp(score_ir) + torch.exp(score_vis) + _DENSE_PRIOR_EPS
                    coef_ir = torch.exp(score_ir) / z
                    coef_vis = torch.exp(score_vis) / z

                fused_map = student(
                    vis=cube_vis,
                    ir=cube_ir,
                    text_features=cap_slice.unsqueeze(0),
                    image_features=roi_slice.unsqueeze(0),
                )

                contrast_target = torch.max(
                    local_contrast_map(cube_ir), local_contrast_map(cube_vis)
                )
                term_texture = _W_CONTRAST * F.l1_loss(
                    local_contrast_map(fused_map), contrast_target
                )

                fg_ir = coef_ir * roi_gate
                fg_vis = coef_vis * roi_gate

                term_ir = pixel_mse(fg_ir * fused_map, fg_ir * cube_ir)
                term_vis = pixel_mse(fg_vis * fused_map, fg_vis * cube_vis)
                term_bg = pixel_mse(fused_map * (1.0 - roi_gate), cube_vis * (1.0 - roi_gate))

                objective = _W_IR * term_ir + _W_VI * term_vis + term_bg + term_texture

                objective.backward()
                solver.step()

                warmup_restarts.step(global_step)
                global_step += 1

                acc_loss_window += objective.item()

                if (step_in_epoch + 1) % cfg.log_loss_interval == 0:
                    mean_in_window = acc_loss_window / cfg.log_loss_interval
                    lr_now = warmup_restarts.get_last_lr()[0]
                    epoch_bar.set_description(
                        f"Epoch {epoch + 1} [{step_in_epoch + 1}/{n_minibatches * minibatch}] "
                        f"Loss: {mean_in_window:.4f}  lr: {lr_now:.2e}"
                    )
                    running_loss_log.append(mean_in_window)
                    acc_loss_window = 0.0

                if (step_in_epoch + 1) % cfg.log_model_interval == 0:
                    student.eval().cpu()
                    fname = f"SDNet_Epoch_{epoch}_iters_{step_in_epoch + 1}.model"
                    torch.save(student.state_dict(), os.path.join(cfg.save_model_dir, fname))

                    mat_file = os.path.join(
                        cfg.save_loss_dir,
                        f"ContentLoss_epoch_{epoch}_iters_{step_in_epoch + 1}.mat",
                    )
                    scio.savemat(mat_file, {"Loss": np.asarray(running_loss_log)})
                    showLossChart(mat_file, os.path.join(cfg.save_loss_dir, "content_loss.png"))

                    student.to(compute_device).train()

                step_in_epoch += 1

    print("\nRight!")
