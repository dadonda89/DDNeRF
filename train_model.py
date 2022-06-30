import os
from nerf.dataset import LlffTrainDataset
from torchvision.transforms import ToTensor
import pdb
import functools
import argparse
import glob
import time
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from poses.pose_utils import load_colmap_data
from poses.colmap_read_model import read_images_binary
from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, architectures,
                  mse2psnr, run_one_iter_of_nerf, load_geometric_data, get_geometric_loss,
                  load_geometric_data_from_sparse_point_cloud, load_depth_maps, run_one_iter_of_mipnerf)
from nerf.train_utils import integrated_pos_enc, get_rays_and_target_for_iter, learning_rate_decay, get_rays_and_target_for_depth_dist
from nerf.data_utils import load_dataset
from utils.visualization import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--load-depth_maps",
        type=str,
        default="",
        help="Path to load saved depth maps from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Load dataset
    images, poses, render_poses, hwf = None, None, None, None
    if cfg.dataset.type.lower() == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )  # hwf = [H, W, F], the camera intrinsic params
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    elif cfg.dataset.type.lower() == "llff":
        adjust_near_far = True

        # if we resume training, near and far in the config already correlate with the normalized poses
        if os.path.exists(configargs.load_checkpoint):
            adjust_near_far = False

        poses, images, H, W, focal, i_train, i_val = load_dataset(cfg, adjust_near_far)
        if cfg.temp.new_dataset:
            train_dataset = LlffTrainDataset(poses[i_train], images[i_train], H, W, focal)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print('device = {}'.format(device))

    encode_position_fn = integrated_pos_enc

    encode_direction_fn = None
    if cfg.models.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.num_encoding_fn_dir,
            include_input=cfg.models.include_input_dir,
            log_sampling=cfg.models.log_sampling_dir,
        )

    # Initialize model.
    model_1 = getattr(architectures, cfg.models.type)(
        num_encoding_fn_xyz=cfg.models.max_ipe_deg,
        num_encoding_fn_dir=cfg.models.num_encoding_fn_dir,
        include_input_xyz=cfg.models.include_input_xyz,
        include_input_dir=cfg.models.include_input_dir,
        use_viewdirs=cfg.models.use_viewdirs,
    )

    model_1.to(device)

    run_one_iter_func = run_one_iter_of_mipnerf
    # Initialize optimizer.
    trainable_parameters = model_1.parameters()

    optimizer_1 = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )
    model_2 = None
    if cfg.temp.two_nets:
        model_2 = architectures.MipNeRFModel(
            num_encoding_fn_xyz=cfg.models.max_ipe_deg,
            num_encoding_fn_dir=cfg.models.num_encoding_fn_dir,
            include_input_xyz=cfg.models.include_input_xyz,
            include_input_dir=cfg.models.include_input_dir,
            use_viewdirs=cfg.models.use_viewdirs,
        )

        model_2.to(device)

        # Initialize optimizer.
        trainable_parameters = model_2.parameters()

        optimizer_2 = getattr(torch.optim, cfg.optimizer.type)(
            trainable_parameters, lr=cfg.optimizer.lr
        )

    lr_function = functools.partial(
      learning_rate_decay,
      lr_init=0.0005,
      lr_final=5e-6,
      max_steps=cfg.experiment.train_iters,
      lr_delay_steps=2500,
      lr_delay_mult=0.01)

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_1.load_state_dict(checkpoint["model_1_state_dict"])
        optimizer_1.load_state_dict(checkpoint["optimizer_1_state_dict"])
        start_iter = checkpoint["iter"] + 1

        if model_2:
            model_2.load_state_dict(checkpoint["model_2_state_dict"])
            optimizer_2.load_state_dict(checkpoint["optimizer_2_state_dict"])

    # load data for depth analysis: will be constant for this training
    img_target = images[i_val[0]].to(device)
    pose_target = poses[i_val[0], :3, :4].to(device)
    da_origins, da_directions, da_rad, da_depth = get_rays_and_target_for_depth_dist(pose_target, H, W, focal,
                                                                                  cfg)

    for i in trange(start_iter, cfg.experiment.train_iters):

        """      
        if cfg.training_methods.depth_is.use and cfg.training_methods.depth_is.reduce_smaples and i == cfg.training_methods.depth_is.reduce_from:

            cfg.training_methods.depth_is.part = 2*cfg.training_methods.depth_is.part
            cfg.nerf.train.num_random_rays = 2*cfg.nerf.train.num_random_rays
            cfg.nerf.train.num_coarse = int(cfg.nerf.train.num_coarse/2)
            cfg.nerf.train.num_fine = int(cfg.nerf.train.num_fine/2)

            with open(os.path.join(logdir, "config_after_reduction.yml"), "w") as f:
                f.write(cfg.dump())  # cfg, f, default_flow_style=False)
        """

        model_1.train()

        lr_new = lr_function(i)

        for param_group in optimizer_1.param_groups:
            param_group["lr"] = lr_new

        if cfg.temp.two_nets:
            model_2.train()
            for param_group in optimizer_2.param_groups:
                param_group["lr"] = lr_new

        img_idx = np.random.choice(i_train)
        img_target = images[img_idx].to(device)
        pose_target = poses[img_idx, :3, :4].to(device)

        if cfg.temp.new_dataset:
            ray_origins, ray_directions, ray_rad, target_s = train_dataset.get_training_rays_for_next_iter(cfg.nerf.train.num_random_rays, device)

        else:
            ray_origins, ray_directions, ray_rad, target_s = get_rays_and_target_for_iter(img_target, pose_target, H, W,
                                                                                          focal, device, cfg)

        if i == cfg.models.top_changes:
            cfg.models.top_k = cfg.models.top_list[1]

        if i == cfg.temp.max_pdf_pad_iters:
            cfg.temp.pdf_padding = False
            print("\npdf padding set to False")

        output = run_one_iter_of_mipnerf(
            H,
            W,
            focal,
            model_1,
            model_2,
            ray_origins,
            ray_directions,
            ray_rad,
            cfg,
            mode="train",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            cycles_number=cfg.models.cycles_number
        )

        loss_list = []
        loss = torch.tensor(0).to(device)

        for j in range(cfg.models.cycles_number):
            loss_list.append(torch.nn.functional.mse_loss(output[j]['rgb'], target_s))
            loss = loss + cfg.models.loss_coeficients[j]*loss_list[j]

        if cfg.models.type == 'DepthMipNeRFModel':

            dp_loss = output[1]["dp_loss"].mean()
            loss += cfg.models.dp_coeficient*dp_loss

            if "raw_rgb_loss" in output[1].keys():
                raw_rgb_loss = output[1]["raw_rgb_loss"].mean()
                loss += cfg.models.raw_rgb_coeficient*raw_rgb_loss

        loss.backward()
        psnr_coarse = mse2psnr(loss_list[0].item())
        psnr_fine = mse2psnr(loss_list[-1].item())
        optimizer_1.step()
        optimizer_1.zero_grad()

        if cfg.temp.two_nets:
            optimizer_2.step()
            optimizer_2.zero_grad()


        # Learning rate updates
        """
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        """


        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                cfg.experiment.id +
                "\n[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr_fine)
            )
        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/coarse_loss", loss_list[0].item(), i)
        if len(loss_list) == 2:
            writer.add_scalar("train/fine_loss", loss_list[1].item(), i)

        if cfg.models.type == 'DepthMipNeRFModel':
            writer.add_scalar("train/depth_prediction_loss", dp_loss, i)
            if "raw_rgb_loss" in output[1].keys():
                writer.add_scalar("train/raw_rgb_loss", raw_rgb_loss, i)

        writer.add_scalar("train/psnr_coarse", psnr_coarse, i)
        writer.add_scalar("train/psnr_fine", psnr_fine, i)
        writer.add_scalar("train/lr", lr_new, i)

        # Validation
        if i > 0 and (i % cfg.experiment.validate_every == 0 or i == cfg.experiment.train_iters - 1):

            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_1.eval()
            if cfg.temp.two_nets:
                model_2.eval()

            start = time.time()
            with torch.no_grad():

                val_idx = int((i/cfg.experiment.validate_every)-1)
                img_idx = i_val[val_idx % len(i_val)] # set i_val
                img_target = images[img_idx].to(device)
                pose_target = poses[img_idx, :3, :4].to(device)
                ray_origins, ray_directions, radii = get_ray_bundle(
                    H, W, focal, pose_target
                )
                output = run_one_iter_func(
                    H,
                    W,
                    focal,
                    model_1,
                    model_2,
                    ray_origins,
                    ray_directions,
                    radii,
                    cfg,
                    mode="validation",
                    encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn,
                    cycles_number=cfg.models.cycles_number
                )

                loss_list = []

                loss = torch.tensor(0).to(device)

                for j in range(cfg.models.cycles_number):
                    loss_list.append(torch.nn.functional.mse_loss(output[j]['rgb'], img_target))
                    loss = loss + cfg.models.loss_coeficients[j] * loss_list[j]

                if cfg.models.type == 'DepthMipNeRFModel':
                    dp_loss = output[1]["dp_loss"].mean()
                    loss += cfg.models.dp_coeficient * dp_loss

                    if "raw_rgb_loss" in output[1].keys():
                        raw_rgb_loss = output[1]["raw_rgb_loss"].mean()
                        loss += cfg.models.raw_rgb_coeficient * raw_rgb_loss

                psnr_fine = mse2psnr(loss_list[-1].item())
                psnr_coarse = mse2psnr(loss_list[0].item())

                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", loss_list[0].item(), i)
                writer.add_scalar("validation/psnr_fine", psnr_fine, i)
                writer.add_scalar("validation/psnr_coarse", psnr_coarse, i)
                writer.add_image("rgb_coarse/coarse", cast_to_image(output[0]["rgb"]), i)
                writer.add_image("disparity_coarse/coarse", cast_to_disparity_image(output[0]["disp"]), i)

                if cfg.models.type == 'DepthMipNeRFModel':
                    writer.add_histogram("depth_prediction/mu_hist", output[0]["mus"].reshape(-1, 1), i)
                    writer.add_histogram("depth_prediction/sigma_hist", output[0]["sigmas"].reshape(-1, 1), i)
                    writer.add_image("disparity_coarse_corr/coarse_corr", cast_to_disparity_image(output[0]['corrected_disp_map']), i)
                    if not cfg.models.val_sampling:
                        writer.add_scalar("validation/depth_prediction_loss", dp_loss, i)
                        if "raw_rgb_loss" in output[1].keys():
                            writer.add_scalar("validation/raw_rgb_loss", raw_rgb_loss, i)

                if "bfp_95" in output[0].keys():
                    writer.add_histogram("bins_per_percentage/coarse_95", output[0]["bfp_95"].reshape(-1, 1), i)

                if "bfp_90" in output[0].keys():
                    writer.add_histogram("bins_per_percentage/coarse_90", output[0]["bfp_90"].reshape(-1, 1), i)

                if "bfp_80" in output[0].keys():
                    writer.add_histogram("bins_per_percentage/coarse_80", output[0]["bfp_80"].reshape(-1, 1), i)

                if "weights_sum" in output[0].keys():
                    writer.add_histogram("weights_sum/coarse", output[0]["weights_sum"].reshape(-1, 1), i)

                if len(loss_list) == 2:
                    writer.add_image("rgb_fine/fine", cast_to_image(output[1]["rgb"]), i)
                    writer.add_image("disparity_fine/fine", cast_to_disparity_image(output[1]["disp"]), i)
                    writer.add_scalar("validation/fine_loss", loss_list[1].item(), i)

                    if cfg.models.type == 'DepthMipNeRFModel' and output[1]['corrected_disp_map']:
                        writer.add_image("disparity_fine_corr/fine_corr", cast_to_disparity_image(output[1]['corrected_disp_map']), i)

                    if "bfp_95" in output[1].keys():
                        writer.add_histogram("bins_per_percentage/fine_95", output[1]["bfp_95"].reshape(-1, 1), i)

                    if "bfp_90" in output[1].keys():
                        writer.add_histogram("bins_per_percentage/fine_90", output[1]["bfp_90"].reshape(-1, 1), i)

                    if "bfp_80" in output[1].keys():
                        writer.add_histogram("bins_per_percentage/fine_80", output[1]["bfp_80"].reshape(-1, 1), i)

                    if "weights_sum" in output[1].keys():
                        writer.add_histogram("weights_sum/fine", output[1]["weights_sum"].reshape(-1, 1), i)

                writer.add_image("rgb/target", cast_to_image(img_target), i)

                if cfg.temp.num_depth_analysis_rays:
                    output = run_one_iter_func(
                        H,
                        W,
                        focal,
                        model_1,
                        model_2,
                        da_origins,
                        da_directions,
                        da_rad,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        cycles_number=cfg.models.cycles_number,
                        depth_analysis_mode=True
                    )

                    for j in range(len(da_depth)):
                        writer.add_image(f"density_distribution_ray_{j}/ray_{j}", get_density_distribution_plots(output, j, da_depth, cfg, i, tb_mode=True), i)


                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr_fine)
                    + " Time: "
                    + str(time.time() - start)
                )

        if i > 0 and (i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1):
            checkpoint_dict = {
                "iter": i,
                "model_1_state_dict": model_1.state_dict(),
                "optimizer_1_state_dict": optimizer_1.state_dict(),
                "loss": loss,
                "psnr": psnr_fine,
            }
            if cfg.temp.two_nets:
                checkpoint_dict["model_2_state_dict"] = model_2.state_dict()
                checkpoint_dict["optimizer_2_state_dict"] = optimizer_2.state_dict()

            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )

    print("Done!")




if __name__ == "__main__":
    from PIL import PngImagePlugin
    LARGE_ENOUGH_NUMBER = 2 * (1024 ** 2)
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER
    main()
