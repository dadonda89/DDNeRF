from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from validation_utils.visualization import *

class Documenter():
    def __init__(self, logdir):

        self.writer = SummaryWriter(logdir)

    def write_train_iter(self, idx, total_loss, loss_list, psnr_coarse, psnr_fine, lr, output, cfg):
        self.writer.add_scalar("train/loss", total_loss.item(), idx)
        self.writer.add_scalar("train/coarse_loss", loss_list[0].item(), idx)
        self.writer.add_scalar("train/fine_loss", loss_list[1].item(), idx)
        if len(loss_list) == 3:
            self.writer.add_scalar("train_depth/depth_prediction_loss", loss_list[2].item(), idx)
            self.writer.add_scalar("train_params/sig_reg_coef", cfg.train_params.sig_regularization, idx)
            self.writer.add_scalar("train_params/gaussian_smooth_factor", cfg.train_params.gaussian_smooth_factor, idx)
            self.writer.add_scalar("train_depth/sig_reg",  output[0]["sig_reg"], idx)
            self.writer.add_scalar("train_depth/sig_loss", output[0]["sig_loss"], idx)
            self.writer.add_scalar("train_depth/mus_reg", output[0]["mus_reg"], idx)
            self.writer.add_scalar("train_depth/mus_loss", output[0]["mus_loss"], idx)

        self.writer.add_scalar("train/psnr_coarse", psnr_coarse, idx)
        self.writer.add_scalar("train/psnr_fine", psnr_fine, idx)
        self.writer.add_scalar("train_params/lr", lr, idx)



    def write_valid_iter(self, idx, total_loss, loss_list, psnr_coarse, psnr_fine, output_dict, img_target, cfg):

        self.writer.add_scalar("validation/loss", total_loss.item(), idx)
        self.writer.add_scalar("validation/coarse_loss", loss_list[0].item(), idx)
        self.writer.add_scalar("validation/psnr_fine", psnr_fine, idx)
        self.writer.add_scalar("validation/psnr_coarse", psnr_coarse, idx)
        self.writer.add_image("rgb_coarse/coarse", cast_to_image(output_dict[0]["rgb"]), idx)
        self.writer.add_image("disparity_coarse/coarse", cast_to_disparity_image(output_dict[0]["disp"]), idx)
        self.writer.add_image("rgb_fine/fine", cast_to_image(output_dict[1]["rgb"]), idx)
        self.writer.add_image("disparity_fine/fine", cast_to_disparity_image(output_dict[1]["disp"]), idx)
        self.writer.add_scalar("validation/fine_loss", loss_list[1].item(), idx)


        self.writer.add_image("rgb/target", cast_to_image(img_target), idx)

        if len(loss_list) == 3:
            self.writer.add_scalar("validation/depth_prediction_loss", loss_list[2].item(), idx)

        if cfg.models.type == 'DDNerfModel':
            self.writer.add_histogram("depth_prediction/mu_hist", output_dict[0]["mus"].reshape(-1, 1), idx)
            self.writer.add_histogram("depth_prediction/sigma_hist", output_dict[0]["sigmas"].reshape(-1, 1), idx)
            self.writer.add_histogram("depth_prediction/smoothed_sigmas", output_dict[0]["smoothed_sigmas"].reshape(-1, 1), idx)
            self.writer.add_image("disparity_coarse_corr/coarse_corr",
                                  cast_to_disparity_image(output_dict[0]['corrected_disp_map']), idx)


    def write_depth_analysis_rays(self, idx, output_dict, da_depth, cfg):

        for j in range(len(da_depth)):
            self.writer.add_image(f"density_distribution_ray_{j}/ray_{j}",
                             get_density_distribution_plots(output_dict, j, da_depth, cfg, idx, tb_mode=True), idx)





