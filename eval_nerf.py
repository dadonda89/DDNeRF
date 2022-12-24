import os
import argparse
import time
import numpy as np
import torch
from data_utils.data_utils import get_datasets
from models import models
import yaml
from tqdm import tqdm
from general_utils import mse2psnr, CfgNode
from validation_utils.visualization import *
from collections import defaultdict
import lpips
from validation_utils.validation import calc_ssim
import imageio
import pickle as pkl

MAX_VALIDATION_IMAGES = 10

def eval_model(basedir, checkpoint_name = "checkpoint", extract_ptc = False, save_images = True):

    config_path = os.path.join(basedir, 'config.yml')
    savedir = os.path.join(basedir, 'validation')
    checkpoint = os.path.join(basedir, f'{checkpoint_name}.ckpt')
    results_file = os.path.join(savedir, "results.txt")

    os.makedirs(savedir, exist_ok=True)


    # Read config file.
    with open(config_path, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # get dataset
    train_dataset, val_dataset = get_datasets(cfg)

    # load data for depth analysis:
    if cfg.train_params.depth_analysis_rays:
        da_origins, da_directions, da_rad, da_depth, da_rgb = val_dataset.get_depth_analysis_rays(device)
        ray_plots_dir = os.path.join(savedir, "rays")
        os.makedirs(ray_plots_dir, exist_ok=True)

    # load model
    model = getattr(models, cfg.nerf.type)(cfg)


    if cfg.train_params.max_pdf_pad_iters < cfg.experiment.train_iters:
        cfg.train_params.pdf_padding = False
        cfg.train_params.gaussian_smooth_factor = cfg.train_params.final_smooth


    checkpoint = torch.load(checkpoint)
    model.load_weights_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    results_dict = defaultdict(dict)
    summary_dict = defaultdict(list)

    if cfg.train_params.depth_analysis_rays:
        with torch.no_grad():
            outputs = model.run_iter(da_origins, da_directions, da_rad, mode="validation",
                                         depth_analysis_validation=True, rgb_target=da_rgb.to(device))

        ray_dict = {}

        for j in range(len(da_depth)):
            img = get_density_distribution_plots(outputs, j, da_depth, cfg)
            imageio.imwrite(os.path.join(ray_plots_dir, f"ray_{j}.png"), img.transpose(1, 2, 0))

            ray_dict["rays"] = defaultdict(dict)
            for rnd in range(2):
                ray_dict["rays"][rnd]["t_vals_for_plot"] = outputs[rnd]["t_vals_for_plot"].cpu()
                ray_dict["rays"][rnd]["uniform_incell_pdf_to_plot"] = outputs[rnd]["uniform_incell_pdf_to_plot"].cpu()

            if 'gaussian_incell_pdf_to_plot' in outputs[1].keys():
                ray_dict["rays"][1]['gaussian_incell_pdf_to_plot'] = outputs[1]['gaussian_incell_pdf_to_plot'].cpu()
                ray_dict["rays"][1]['smoothed_gaussian_incell_pdf_to_plot'] = outputs[1]['smoothed_gaussian_incell_pdf_to_plot'].cpu()

            ray_dict["gt_depth"] = da_depth

        with open(os.path.join(ray_plots_dir, f"ray_dict.pkl"), "wb") as f:
            pkl.dump(ray_dict, f)


    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores

    # Evaluation loop
    model_time_per_image = []
    for i in range(min(len(val_dataset.poses), MAX_VALIDATION_IMAGES)):

        pose = val_dataset.poses[i]
        save_path = os.path.join(savedir, f"val_image_{i + 1}")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "pose.npy"), pose.numpy())

        start = time.time()

        with torch.no_grad():
            ray_origins, ray_directions, radii, img_target = val_dataset.get_next_validation_rays(device)

            outputs = model.run_iter(ray_origins, ray_directions, radii, mode="validation",
                                    depth_analysis_validation=False, rgb_target=img_target)

        model_time_per_image.append(time.time() - start)

        if extract_ptc:
            xyz = (ray_directions * outputs[1]["depth"].unsqueeze(-1) + ray_origins).cpu().numpy()


        save_path = os.path.join(savedir, f"val_image_{i+1}")
        os.makedirs(save_path, exist_ok=True)

        if extract_ptc:
            savefile = os.path.join(save_path, f"xyz.npy")
            np.save(savefile, xyz)

        if save_images:
            save_validation_images(outputs, save_path)

        # PSNR Calculations:
        psnr_coarse = mse2psnr(torch.nn.functional.mse_loss(outputs[0]['rgb'], img_target))
        psnr_fine = mse2psnr(torch.nn.functional.mse_loss(outputs[1]['rgb'], img_target))

        summary_dict["psnr_coarse"].append(psnr_coarse)
        summary_dict["psnr_fine"].append(psnr_fine)

        results_dict[i]["psnr_coarse"] = psnr_coarse
        results_dict[i]["psnr_fine"] = psnr_fine

        #LPIPS calculations:
        # [0,1] normalization H,W,3 shape to [-1,1] normalization 1,3,H,W shape
        coarse_rgb_lpips = ((outputs[0]['rgb'].permute(2, 0, 1).unsqueeze(0).cpu()-0.5) * 2).to(torch.float32)
        fine_rgb_lpips = ((outputs[1]['rgb'].permute(2, 0, 1).unsqueeze(0).cpu() - 0.5) * 2).to(torch.float32)
        gt_rgb_lpips = ((img_target.permute(2, 0, 1).unsqueeze(0).cpu() - 0.5) * 2).to(torch.float32)


        lpips_coarse = float(loss_fn_alex(coarse_rgb_lpips, gt_rgb_lpips).squeeze())
        lpips_fine = float(loss_fn_alex(fine_rgb_lpips, gt_rgb_lpips).squeeze())

        summary_dict["lpips_coarse"].append(lpips_coarse)
        summary_dict["lpips_fine"].append(lpips_fine)

        results_dict[i]["lpips_coarse"] = lpips_coarse
        results_dict[i]["lpips_fine"] = lpips_fine

        # ssim calculation:
        results_dict[i]["ssim_coarse_v1"], results_dict[i]["ssim_coarse_v2"] = calc_ssim(outputs[0]['rgb'], img_target)
        results_dict[i]["ssim_fine_v1"], results_dict[i]["ssim_fine_v2"] = calc_ssim(outputs[1]['rgb'], img_target)

        summary_dict["ssim_coarse_v1"].append(results_dict[i]["ssim_coarse_v1"])
        summary_dict["ssim_fine_v1"].append(results_dict[i]["ssim_fine_v1"])
        summary_dict["ssim_coarse_v2"].append(results_dict[i]["ssim_coarse_v2"])
        summary_dict["ssim_fine_v2"].append(results_dict[i]["ssim_fine_v2"])


        tqdm.write(f"Avg time per image: {sum(model_time_per_image) / (i + 1)}")

    write_dicts_to_a_file(summary_dict, results_dict, results_file)


if __name__ == "__main__":
    from PIL import PngImagePlugin
    LARGE_ENOUGH_NUMBER = 2
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="Path to experiment log dir.")
    parser.add_argument("--checkpoint", type=str, default='checkpoint', help="Path to experiment log dir.")
    parser.add_argument("--save_images", type=bool, default=True, help="denote if to save images one by one of only video")
    parser.add_argument("--extract_ptc", type=bool, default=False, help="denote if to extract point cloud from images")

    configargs = parser.parse_args()

    eval_model(configargs.logdir, checkpoint_name=configargs.checkpoint, extract_ptc=configargs.extract_ptc, save_images=configargs.save_images)
