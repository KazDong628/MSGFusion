import gc
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import third_party_clip.clip as clip_backend

from msgfusion.config_shim import runtime_config as cfg
from msgfusion.models.fusion_network import build_msgfusion_network


def run_llvip_benchmark() -> None:
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_net, _ = clip_backend.load("ViT-B/32", device=runtime_device)

    cap_file = "../embeddings/caption_test/caption_embeddings_test_LLVIP.pt"
    vis_file = "../embeddings/visual_test/visual_embedding_test_LLVIP.pt"
    if not (os.path.exists(cap_file) and os.path.exists(vis_file)):
        raise FileNotFoundError("Required embedding files not found.")

    caption_tensor = torch.load(cap_file).to(runtime_device)
    visual_tensor = torch.load(vis_file).to(runtime_device)

    data_root = "../IVT_test_datasets/IVT_test_LLVIP"
    weights_fp = cfg.model_path_gray
    print("Test Model:" + weights_fp)

    with torch.no_grad():
        fusion_net = _wrap_dataparallel(weights_fp, runtime_device)
        export_dir = "outputs_llvip/"
        os.makedirs(export_dir, exist_ok=True)

        for idx in range(250):
            sample_id = idx + 1
            ir_fp = os.path.join(data_root, "ir", f"{sample_id}.png")
            rgb_fp = os.path.join(data_root, "vis", f"{sample_id}.png")
            if not (os.path.exists(ir_fp) and os.path.exists(rgb_fp)):
                print(f"Skipping index {sample_id} because image files are missing.")
                continue
            _fuse_one_pair(
                runtime_device,
                fusion_net,
                ir_fp,
                rgb_fp,
                export_dir,
                caption_tensor,
                visual_tensor,
                sample_id,
            )

    print("Done......")


def _wrap_dataparallel(path: str, device: str) -> torch.nn.Module:
    net = build_msgfusion_network()
    net.load_state_dict(torch.load(path, map_location=device))
    wrapped = torch.nn.DataParallel(net, device_ids=[0])
    wrapped.eval()
    return wrapped


def _rgb_to_ycbcr(image: Image.Image):
    rgb_array = np.array(image)
    rgb2y = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
    ycrcb = np.dot(rgb_array, rgb2y.T)
    return ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]


def _ycbcr_to_rgb(y_plane, cb_plane, cr_plane) -> Image.Image:
    ycrcb_array = np.stack((y_plane, cb_plane, cr_plane), axis=-1)
    yuv2rgb = np.array([[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]])
    rgb_array = np.dot(ycrcb_array, yuv2rgb.T)
    rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb_array, mode="RGB")


def _fuse_one_pair(device, model, ir_path, vis_path, out_root, captions, visuals, sample_no):
    ir01 = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE) / 255.0
    rgb_pil = Image.open(vis_path).convert("RGB")
    y_plane, cb_plane, cr_plane = _rgb_to_ycbcr(rgb_pil)
    vis_gray01 = y_plane / 255.0

    ir_t = torch.from_numpy(ir01).float().unsqueeze(0).unsqueeze(0).to(device)
    vis_t = torch.from_numpy(vis_gray01).float().unsqueeze(0).unsqueeze(0).to(device)

    cap_slice = captions[(sample_no - 1) * 5 : sample_no * 5].unsqueeze(0)
    vis_slice = visuals[(sample_no - 1) * 5 : sample_no * 5].unsqueeze(0)

    prediction = model(vis=vis_t, ir=ir_t, text_features=cap_slice, image_features=vis_slice)

    fused_gray = (prediction.cpu().numpy()[0][0] * 255).clip(0, 255)
    rgb_out = _ycbcr_to_rgb(fused_gray, cb_plane, cr_plane)

    dst = os.path.join(out_root, f"fused_{sample_no}.jpg")
    rgb_out.save(dst)
    print(f"Saved: {dst}")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    run_llvip_benchmark()
