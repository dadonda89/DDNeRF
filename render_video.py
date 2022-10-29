import os
import argparse
import time
import imageio
import numpy as np
import torch
import torchvision
import yaml
import cv2
from tqdm import tqdm
from general_utils import CfgNode, get_ray_bundle
from validation_utils.visualization import *
from data_utils.load_blender import pose_spherical_for_real_world_360
from data_utils.data_utils import load_dataset, get_datasets
from models import models
import pathlib



def render_model_video(basedir, save_images = True):

    config_path = os.path.join(basedir, 'config.yml')
    savedir = os.path.join(basedir, 'video')
    checkpoint = os.path.join(basedir, 'checkpoint.ckpt')

    os.makedirs(savedir, exist_ok=True)

    # Read config file.
    with open(config_path, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # load model
    model = getattr(models, cfg.nerf.type)(cfg)

    # this try - except is because of config format changes

    if cfg.train_params.max_pdf_pad_iters < cfg.experiment.train_iters:
        cfg.train_params.pdf_padding = False
        cfg.train_params.gaussian_smooth_factor = cfg.train_params.final_smooth


    checkpoint = torch.load(checkpoint)
    model.load_weights_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    # get dataset
    train_dataset, val_dataset = get_datasets(cfg)

    # init video writer:
    fps = 24
    video_path = os.path.join(savedir, "video")
    os.makedirs(video_path, exist_ok=True)

    file_name = video_path + '/' + cfg.experiment.id + '.avi'

    H, W, focal = val_dataset.H, val_dataset.W, val_dataset.focal

    writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (int(2 * W), int(H)))

    # Create directory to save images to.
    os.makedirs(savedir, exist_ok=True)

    os.makedirs(os.path.join(savedir, "disparity"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "images"), exist_ok=True)

    # Evaluation loop
    times_per_image = []
    img_num = 120
    for i in tqdm(range(img_num)):

        start = time.time()
        with torch.no_grad():
            ray_origins, ray_directions, radii = val_dataset.get_next_render_pose(device)
            output = model.run_iter(ray_origins, ray_directions, radii, mode="validation", depth_analysis_validation=False)

            rgb = output[1]["rgb"]

            disp = output[1]["disp"]
            disp = cast_to_disparity_image(disp).squeeze()

        times_per_image.append(time.time() - start)

        if save_images:
            image_savedir = os.path.join(savedir, 'images')

            savefile = os.path.join(image_savedir, f"{i:04d}.png")
            imageio.imwrite(savefile, cast_to_image(rgb[..., :3]).transpose(1,2,0))

            savefile = os.path.join(savedir, "disparity", f"{i:04d}.png")
            imageio.imwrite(savefile, disp)

        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")

        rgb = 255*rgb[..., :3].permute(2, 0, 1).cpu().detach()
        disp = torch.from_numpy(disp).expand(3, disp.shape[0], disp.shape[1])
        frame = torch.cat((rgb, disp), dim=2).numpy().astype(np.uint8)

        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    from PIL import PngImagePlugin
    LARGE_ENOUGH_NUMBER = 2
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)
    base_dir = "/home/ddadon/nerf/DDNeRF/logs/final_code/dd_4_256"
    render_model_video(base_dir, save_images = False)
    base_dir = "/home/ddadon/nerf/DDNeRF/logs/final_code/r_4_256"
    render_model_video(base_dir, save_images=False)




