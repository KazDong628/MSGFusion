"""
Experiment defaults for MSGFusion (paths, schedule, patch geometry).

Field names stay stable: training and data I/O modules bind via
``msgfusion.config_shim.runtime_config`` (or ``args`` alias).
"""


class FusionTrainingConfig:
    """Static container of scalar/list options (no instances required)."""

    image_size = 256
    HEIGHT = 256
    WIDTH = 256
    PATCH_SIZE = 128
    PATCH_STRIDE = 4

    trainNumber = 2000
    batch_size = 1
    epochs = 3

    lr = 1e-4
    lr_light = 1e-4
    log_loss_interval = 5
    log_model_interval = 500

    cuda = 1
    device = 0

    save_model_dir = "models"
    save_loss_dir = "models/loss"

    model_path_gray = "../models/msgfusion_best.model"

    mask_dir = "/root/lanyun-tmp/dataset/mask"
    lambda_bg = 1.0
    lambda_struct = 0.05
    lambda_color = 0.0005
    ssim_weight = [1, 10, 100, 1000, 10000]
    ssim_path = ["1e0", "1e1", "1e2", "1e3", "1e4"]
