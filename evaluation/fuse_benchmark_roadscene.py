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


def run_roadscene_benchmark() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip, _ = clip_backend.load("ViT-B/32", device=device)

    cap_embeds_path = "../embeddings/caption_test/caption_embeddings_test_RoadScene.pt"
    visual_embeds_path = "../embeddings/visual_test/visual_embedding_test_RoadScene.pt"

    if not os.path.exists(cap_embeds_path) or not os.path.exists(visual_embeds_path):
        raise FileNotFoundError("Required embedding files not found.")

    cap_embeds = torch.load(cap_embeds_path).to(device)
    visual_embeds = torch.load(visual_embeds_path).to(device)

    test_path = "../IVT_test_datasets/IVT_test_RoadScene"
    model_path = cfg.model_path_gray
    print("Test Model:" + model_path)
    with torch.no_grad():
        model = _wrap_dataparallel(model_path, device)
        output_path = "outputs_roadscene/"
        os.makedirs(output_path, exist_ok=True)

        for i in range(50):
            index = i + 1

            infrared_path = os.path.join(test_path, f"ir/{index}.png")
            visible_path = os.path.join(test_path, f"vis/{index}.png")

            if not os.path.exists(infrared_path) or not os.path.exists(visible_path):
                print(f"Skipping index {index} because image files are missing.")
                continue

            _fuse_one_sample(device, model, infrared_path, visible_path, output_path, cap_embeds, visual_embeds, index)

    print("Done......")


def _wrap_dataparallel(path: str, device: str) -> torch.nn.Module:
    model = build_msgfusion_network()
    model.load_state_dict(torch.load(path, map_location=device))
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.eval()
    return model


def _rgb_to_ycbcr(image):
    rgb_array = np.array(image)
    transform_matrix = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
    ycbcr_array = np.dot(rgb_array, transform_matrix.T)
    return ycbcr_array[:, :, 0], ycbcr_array[:, :, 1], ycbcr_array[:, :, 2]


def _ycbcr_to_rgb(y, cb, cr):
    ycbcr_array = np.stack((y, cb, cr), axis=-1)
    transform_matrix = np.array([[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]])
    rgb_array = np.dot(ycbcr_array, transform_matrix.T)
    rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb_array, mode="RGB")


def _fuse_one_sample(device, model, infrared_path, visible_path, output_path_root, cap_embeds, visual_embeds, idx):
    ir_img = cv2.imread(infrared_path, cv2.IMREAD_GRAYSCALE) / 255.0
    vi_img = Image.open(visible_path).convert("RGB")
    vi_img_y, vi_img_cb, vi_img_cr = _rgb_to_ycbcr(vi_img)
    vi_img = vi_img_y / 255.0

    ir_tensor = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0).to(device)
    vi_tensor = torch.from_numpy(vi_img).float().unsqueeze(0).unsqueeze(0).to(device)

    text_features = cap_embeds[(idx - 1) * 5 : idx * 5].unsqueeze(0)
    visual_features = visual_embeds[(idx - 1) * 5 : idx * 5].unsqueeze(0)

    output = model(vis=vi_tensor, ir=ir_tensor, text_features=text_features, image_features=visual_features)

    fused_image = (output.cpu().numpy()[0][0] * 255).clip(0, 255)
    fused_image = _ycbcr_to_rgb(fused_image, vi_img_cb, vi_img_cr)

    out_fp = os.path.join(output_path_root, f"fused_{idx}.jpg")
    fused_image.save(out_fp)
    print(f"Saved: {out_fp}")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    run_roadscene_benchmark()
