import os
from data_utils.dataset_helpers import switch_t_ndc_to_regular
from data_utils.data_utils import get_datasets
from models import models
from validation_utils.documentation import Documenter
import functools
import argparse
import time
import numpy as np
import torch
import yaml
from tqdm import tqdm, trange

from general_utils.cfgnode import CfgNode
from general_utils.nerf_helpers import learning_rate_decay, mse2psnr



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

    configargs = parser.parse_args()

    # Read config file.
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    doc = Documenter(logdir)

    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  #

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

    # get datasets
    train_dataset, val_dataset = get_datasets(cfg)

    # load data for depth analysis:
    if cfg.train_params.depth_analysis_rays:
        da_origins, da_directions, da_rad, da_depth, da_rgb = val_dataset.get_depth_analysis_rays(device)


    # load model
    model = getattr(models, cfg.nerf.type)(cfg)
    model.to(device)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model.load_weights_from_checkpoint(checkpoint)
        start_iter = checkpoint["iter"] + 1
        val_dataset.current_idx = (checkpoint["iter"]//cfg.experiment.validate_every) % (val_dataset.images.shape[0])

    # Initialize optimizer.
    optims = []

    trainable_parameters = model.coarse.parameters()

    optims.append(getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    ))

    # add optimizer for DDNeRF fine model
    if cfg.nerf.type != "GeneralMipNerfModel":
        trainable_parameters = model.fine.parameters()

        optims.append(getattr(torch.optim, cfg.optimizer.type)(
            trainable_parameters, lr=cfg.optimizer.lr
        ))

    # set lr function
    lr_function = functools.partial(
      learning_rate_decay,
      lr_init=0.0005,
      lr_final=5e-6,
      max_steps=cfg.experiment.train_iters,
      lr_delay_steps=2500,
      lr_delay_mult=0.01)


    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        for i, optim in enumerate(optims):

            optim.load_state_dict(checkpoint[f"optimizer_{i+1}_state_dict"])


        if start_iter > cfg.train_params.max_pdf_pad_iters:
            model.cfg.train_params.pdf_padding = False

    # init training adjustable parameters
    dsmooth = (cfg.train_params.gaussian_smooth_factor - cfg.train_params.final_smooth)/cfg.train_params.finnish_smooth
    initial_gaussian_smooth = cfg.train_params.gaussian_smooth_factor

    if cfg.train_params.set_automatic_dist_reg_coeficient:
        cfg.train_params.dist_reg_coeficient = min(max(1/cfg.nerf.train.num_coarse, 0.01), 0.12)
    print(f"set dist_reg_coeficient to - {cfg.train_params.dist_reg_coeficient}")

    ####################
    ### training loop###
    ####################

    for i in trange(start_iter, cfg.experiment.train_iters):

        # adjust smoothness factors to the iteration number
        if (i < cfg.train_params.finnish_smooth):
            model.cfg.train_params.gaussian_smooth_factor = initial_gaussian_smooth - dsmooth*i
        else:
            model.cfg.train_params.gaussian_smooth_factor = cfg.train_params.final_smooth

        if i == cfg.train_params.max_pdf_pad_iters:
            model.cfg.train_params.pdf_padding = False
            print("\npdf padding set to False")

        model.train()

        lr_new = lr_function(i)

        for optimizer in optims:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_new

        ray_origins, ray_directions, ray_rad, target_s = train_dataset.get_training_rays_for_next_iter(cfg.nerf.train.num_random_rays, device)

        output = model.run_iter(ray_origins, ray_directions, ray_rad, mode="train", depth_analysis_validation=False, rgb_target=target_s)

        loss_list = []
        loss = torch.tensor(0).to(device)

        for j in range(len(output)):
            loss_list.append(torch.nn.functional.mse_loss(output[j]['rgb'], target_s))
            loss = loss + cfg.train_params.loss_coeficients[j]*loss_list[j]

        if cfg.nerf.type == 'DDNerfModel':

            dp_loss = output[1]["dp_loss"].mean()
            loss += cfg.train_params.dp_coeficient*dp_loss
            loss_list.append(dp_loss)


        loss.backward()
        psnr_coarse = mse2psnr(loss_list[0].item())
        psnr_fine = mse2psnr(loss_list[1].item())


        for optimizer in optims:
            optimizer.step()
            optimizer.zero_grad()


        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                cfg.experiment.id +
                "\n[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr_fine)
                + " dp coef: "
                + str(cfg.train_params.dp_coeficient)
            )
        doc.write_train_iter(i, loss, loss_list, psnr_coarse, psnr_fine, lr_new, output, model.cfg)
        #######################
        ##### Validation ######
        #######################

        if i > -1 and (i % cfg.experiment.validate_every == 0 or i == cfg.experiment.train_iters - 1):

            tqdm.write("[VAL] =======> Iter: " + str(i))
            model.eval()

            start = time.time()
            with torch.no_grad():

                ray_origins, ray_directions, radii, img_target = val_dataset.get_next_validation_rays(device)

                output = model.run_iter(ray_origins, ray_directions, radii, mode="validation", depth_analysis_validation=False, rgb_target=img_target)

                loss_list = []

                loss = torch.tensor(0).to(device)

                for j in range(len(output)):
                    loss_list.append(torch.nn.functional.mse_loss(output[j]['rgb'], img_target))
                    loss = loss + cfg.train_params.loss_coeficients[j] * loss_list[j]

                if cfg.nerf.type == 'DDNerfModel':
                    dp_loss = output[1]["dp_loss"].mean()
                    loss += cfg.train_params.dp_coeficient * dp_loss
                    loss_list.append(dp_loss)

                psnr_fine = mse2psnr(loss_list[1].item())
                psnr_coarse = mse2psnr(loss_list[0].item())

                if cfg.dataset.ndc_rays:
                    ro, rd, _ = val_dataset.get_current_regular_validation_rays(device)
                    output[0]["depth"] = switch_t_ndc_to_regular(output[0]["depth"], ro, rd)
                    output[1]["depth"] = switch_t_ndc_to_regular(output[1]["depth"], ro, rd)

                doc.write_valid_iter(i, loss, loss_list, psnr_coarse, psnr_fine, output, img_target, model.cfg)

                if cfg.train_params.depth_analysis_rays:
                    output_dict = model.run_iter(da_origins, da_directions, da_rad, mode="validation",
                                                 depth_analysis_validation=True, rgb_target=da_rgb.to(device))

                    doc.write_depth_analysis_rays(i, output_dict, da_depth, model.cfg)

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
                "model_1_state_dict": model.coarse.state_dict(),
                "optimizer_1_state_dict": optims[0].state_dict(),
                "loss": loss,
                "psnr": psnr_fine
            }
            if cfg.nerf.type != "GeneralMipNerfModel":
                checkpoint_dict["model_2_state_dict"] = model.fine.state_dict()
                checkpoint_dict["optimizer_2_state_dict"] = optims[1].state_dict()

            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint.ckpt"),
            )
    print("Done!")


if __name__ == "__main__":
    from PIL import PngImagePlugin
    LARGE_ENOUGH_NUMBER = 2
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)
    main()
