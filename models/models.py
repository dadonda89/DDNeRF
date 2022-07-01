import torch

from models import base_architectures
from models.samplers_nad_helpers import *
from nerf_utils import get_embedding_function
from nerf_utils.nerf_helpers_and_samplers import get_minibatches
from nerf_utils.volume_rendering_utils import volume_render_radiance_field
from nerf_utils.math_utils import *
from nerf_utils.loss import *
from nerf_utils import run_network

class GeneralMipNerfModel(torch.nn.Module):
    """
    this class represent general mip nerf model
    """

    def __init__(self, cfg, backbone="MipNeRFModel"):
        super(GeneralMipNerfModel, self).__init__()

        self.coarse = getattr(base_architectures, backbone)(
                        max_ipe_deg=16,
                        num_encoding_fn_dir=4,
                        include_input_xyz=False,
                        include_input_dir=True,
                        use_viewdirs=True,
                    )
        self.fine = self.coarse


        self.encode_position_fn = integrated_pos_enc
        self.cfg = cfg

        self.encode_direction_fn = get_embedding_function(
            num_encoding_functions=4,
            include_input=True,
            log_sampling=True,
        )

    def run_iter(self, ray_origins, ray_directions, ray_rad, mode="train", depth_analysis_validation=False, rgb_target=None):

        restore_shapes_dict = {}
        restore_shapes_dict['rgb'] = ray_directions.shape
        restore_shapes_dict['depth'] = ray_directions.shape[:-1]

        batches = self.get_rays_batches(ray_origins, ray_directions, ray_rad, mode)

        if rgb_target is not None:
            rgb_targets = get_minibatches(rgb_target.view((-1, 3)), chunksize=getattr(self.cfg.nerf, mode).chunksize)
        else:
            rgb_targets = [None for batch in batches]

        pred = [self.predict(batch, mode, depth_analysis_validation, rgb_target) for batch, rgb_target in zip(batches, rgb_targets)]

        output = pred[0]

        for i in range(1, len(pred)):
            for j in range(len(output)):
                for key in pred[i][j].keys():
                    if (pred[i][j][key] is not None) and (pred[i][j][key] is not False):
                        output[j][key] = torch.cat((output[j][key], pred[i][j][key]), dim=0)

        # reshape for validation
        if mode == "validation" and not depth_analysis_validation:

            for i in range(len(output)):
                output[i]["rgb"] = output[i]["rgb"].view(restore_shapes_dict['rgb'])
                output[i]["disp"] = output[i]["disp"].view(restore_shapes_dict['depth'])
                output[i]["acc"] = output[i]["acc"].view(restore_shapes_dict['depth'])
                output[i]["depth"] = output[i]["depth"].view(restore_shapes_dict['depth'])
                if ("corrected_disp_map" in output[i].keys()) and (output[i]["corrected_disp_map"] is not None):
                    output[i]["corrected_disp_map"] = output[i]["corrected_disp_map"].view(restore_shapes_dict['depth'])
        return output

    def predict(self, ray_batch, mode, depth_analysis_validation, rgb_target = None):

        ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
        rr = ray_batch[..., 6].reshape(-1, 1)
        bounds = ray_batch[..., 7:9].view((-1, 1, 2))
        near, far = bounds[..., 0], bounds[..., 1]

        ret_dict = {}

        for i in range(2):
            if i == 0:
                t_vals = sample_first_cycle(self.cfg, near, far, mode)
            else:
                t_vals = sample_pdf(
                    t_vals,
                    weights,
                    getattr(self.cfg.nerf, mode).num_fine + 1,
                    self.cfg,
                    det=(getattr(self.cfg.nerf, mode).perturb == 0.0),
                )
                t_vals = t_vals.detach()

            samples = cast_rays(t_vals, ro, rd, rr, self.cfg.nerf.ray_shape)

            enc_samples = self.encode_position_fn(samples)

            radiance_field = run_network(self.coarse, enc_samples, ray_batch, getattr(self.cfg.nerf, mode).chunksize,
                                         self.encode_direction_fn)

            rgb, disp, acc, weights, depth, _, _ = volume_render_radiance_field(radiance_field, t_vals, rd,
                                                                                radiance_field_noise_std=getattr(
                                                                                    self.cfg.nerf,
                                                                                    mode).radiance_field_noise_std,
                                                                                white_background=getattr(self.cfg.nerf,
                                                                                            mode).white_background,
                                                                                cfg=self.cfg
                                                                                )
            bfp_95 = bins_for_percentage(weights, 0.95)
            bfp_90 = bins_for_percentage(weights, 0.9)
            bfp_80 = bins_for_percentage(weights, 0.8)

            ret_dict[i] = {"rgb": rgb, "disp": disp, "acc": acc, "weights": weights, "depth": depth,
                           "bfp_95": bfp_95, "bfp_90": bfp_90, "bfp_80": bfp_80}

            if depth_analysis_validation:
                ret_dict[i]["uniform_incell_pdf_to_plot"] = get_uniform_incell_pdf(t_vals, weights, self.cfg)
                ret_dict[i]["t_vals_for_plot"] = t_vals

        return ret_dict

    def get_rays_batches(self, ray_origins, ray_directions, ray_rad, mode):

        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))

        ro = ray_origins.view((-1, 3))
        rd = ray_directions.reshape((-1, 3))
        ray_rad = ray_rad.reshape((-1, 1))

        near = self.cfg.dataset.near * torch.ones_like(rd[..., :1])
        far = self.cfg.dataset.far * torch.ones_like(rd[..., :1])

        rays = torch.cat((ro, rd, ray_rad, near, far, viewdirs), dim=-1)

        batches = get_minibatches(rays, chunksize=getattr(self.cfg.nerf, mode).chunksize)

        return batches

    def to(self, device):

        self.coarse.to(device)
        self.fine.to(device)

    def load_weights_from_checkpoint(self, checkpoint):

        self.coarse.load_state_dict(checkpoint["model_1_state_dict"])
        if self.cfg.models.type != "GeneralMipNerfModel":
            self.fine.load_state_dict(checkpoint["model_2_state_dict"])


    def train(self):

        self.coarse.train()
        self.fine.train()

    def eval(self):

        self.coarse.eval()
        self.fine.eval()


class DDNerfModel(GeneralMipNerfModel):

    def __init__(self, cfg):
        GeneralMipNerfModel.__init__(self, cfg, backbone="DepthMipNeRFModel")

        self.fine = base_architectures.MipNeRFModel(
            max_ipe_deg=16,
            num_encoding_fn_dir=4,
            include_input_xyz=False,
            include_input_dir=True,
            use_viewdirs=True,
        )

    def predict(self, ray_batch, mode, depth_analysis_validation, rgb_target):

        ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
        rr = ray_batch[..., 6].reshape(-1, 1)
        bounds = ray_batch[..., 7:9].view((-1, 1, 2))
        near, far = bounds[..., 0], bounds[..., 1]

        ret_dict = {}

        model = self.coarse

        for i in range(2):

            if i == 1:
                model = self.fine
                mus = None

            if i == 0:
                t_vals = sample_first_cycle(self.cfg, near, far, mode)

            else:
                t_vals = sample_pdf_with_mu_sigma(
                    t_vals,
                    weights,
                    mus_0,
                    smoothed_sigmas,
                    smoothed_part_inside_bins,
                    smoothed_left_tail,
                    getattr(self.cfg.nerf, mode).num_fine + 1,
                    self.cfg,
                    det=(getattr(self.cfg.nerf, mode).perturb == 0.0)
                )

            samples = cast_rays(t_vals, ro, rd, rr, self.cfg.nerf.ray_shape)

            enc_samples = self.encode_position_fn(samples)

            radiance_field = run_network(model, enc_samples, ray_batch, getattr(self.cfg.nerf, mode).chunksize,
                                         self.encode_direction_fn)

            if i == 0:
                raw_mus, raw_sigmas = radiance_field[:, :, -2], radiance_field[:, :, -1]

                sig_loss = torch.nn.functional.mse_loss(raw_sigmas, torch.zeros_like(raw_sigmas))
                mus_loss = torch.nn.functional.mse_loss(raw_mus, torch.zeros_like(raw_mus))

                mus = torch.sigmoid(raw_mus / self.cfg.train_params.reg_reduction_factor)
                sigmas = torch.sigmoid(raw_sigmas / self.cfg.train_params.reg_reduction_factor) + 0.001
                #sigmas = sigmas*0 + 0.25

                try:
                    if self.cfg.train_params.musig_activation == "sin":
                        mus = (torch.sin(raw_mus)+1)/2
                        sigmas = (torch.sin(raw_sigmas)+1)/2 + 0.001

                except:
                    pass

                try:
                    if self.cfg.train_params.reg_loss_type == "L1":
                        sig_loss = torch.nn.functional.l1_loss(raw_sigmas, torch.zeros_like(raw_sigmas))
                        mus_loss = torch.nn.functional.l1_loss(raw_mus, torch.zeros_like(raw_mus))

                    elif self.cfg.train_params.reg_loss_type == "MSE_v1":
                        sig_loss = (torch.abs(raw_sigmas)**self.cfg.train_params.p).sum()/raw_sigmas.shape[0]
                        mus_loss = (torch.abs(raw_mus)**self.cfg.train_params.p).sum()/raw_mus.shape[0]


                    elif self.cfg.train_params.reg_loss_type == "MSE_v2":
                        sig_loss = torch.nn.functional.mse_loss(sigmas, torch.ones_like(sigmas)*0.5, reduction="sum")
                        mus_loss = torch.nn.functional.mse_loss(mus, torch.ones_like(mus)*0.5, reduction="sum")

                    elif self.cfg.train_params.reg_loss_type == "MSE_v3":
                        sig_loss = torch.nn.functional.mse_loss(sigmas, torch.ones_like(sigmas)*0.5)*self.cfg.nerf.train.num_coarse
                        mus_loss = torch.nn.functional.mse_loss(mus, torch.ones_like(mus)*0.5)*self.cfg.nerf.train.num_coarse

                except:
                    pass

                mus_reg = self.cfg.train_params.mu_regularization * mus_loss
                sig_reg = self.cfg.train_params.sig_regularization * sig_loss

                x_0 = (0 - mus) / sigmas
                x_1 = (1 - mus) / sigmas

                left_tail = approximate_cdf(x_0)
                part_inside_bins = (approximate_cdf(x_1) - left_tail)




                radiance_field = radiance_field[:, :, :-2]

            rgb, disp, acc, weights, depth, corrected_disp_map, rgb_raw = volume_render_radiance_field(
                radiance_field, t_vals, rd, radiance_field_noise_std=getattr(self.cfg.nerf, mode).radiance_field_noise_std,
                           white_background=getattr(self.cfg.nerf, mode).white_background, mus=mus,  cfg=self.cfg)

            if i == 0:
                # smoothing in-cell distribution before re-sampling
                smoothed_sigmas = sigmas * self.cfg.train_params.gaussian_smooth_factor

                if rgb_target is not None and self.cfg.train_params.local_smooth_factor:

                    local_smooth = torch.exp(self.cfg.train_params.local_smooth_factor*(((rgb_target - rgb) ** 2).sum(-1)).reshape(-1, 1))
                    local_smooth = torch.clip(local_smooth, 1, self.cfg.train_params.max_local_valus)
                    smoothed_sigmas = smoothed_sigmas*local_smooth


                x_0 = (0 - mus) / smoothed_sigmas
                x_1 = (1 - mus) / smoothed_sigmas
                smoothed_left_tail = approximate_cdf(x_0)
                smoothed_part_inside_bins = (approximate_cdf(x_1) - smoothed_left_tail)

            bfp_95 = bins_for_percentage(weights, 0.95)
            bfp_90 = bins_for_percentage(weights, 0.9)
            bfp_80 = bins_for_percentage(weights, 0.8)

            if i == 0:
                t_vals_0 = t_vals
                mus_0 = mus
                sigmas_0 = sigmas
                weights_0 = weights
                left_tails_0 = left_tail
                part_inside_cells_0 = part_inside_bins

            dp_loss = None
            if i == 1:

                dp_loss = estimate_dp_loss_v6(t_vals.detach(), t_vals_0.detach(), weights.detach(), weights_0, mus_0,
                                              sigmas_0, left_tails_0.detach(), part_inside_cells_0.detach(), self.cfg)*(t_vals.shape[1]-1)
                dp_loss = (dp_loss + mus_reg + sig_reg).unsqueeze(0)


            if mus is not None:
                pdf = (weights / torch.sum(weights, dim=-1, keepdim=True))
                mus_to_record = mus[pdf > 0.1]
                sigmas_to_record = sigmas[pdf > 0.1]

            ret_dict[i] = {"rgb": rgb, "disp": disp, "acc": acc, "weights": weights, "depth": depth,
                           "mus": mus_to_record,
                           "sigmas": sigmas_to_record, "dp_loss": dp_loss, "corrected_disp_map": corrected_disp_map,
                           "bfp_95": bfp_95, "bfp_90": bfp_90, "bfp_80": bfp_80, "weights_sum": weights.sum(-1), "smoothed_sigmas" : smoothed_sigmas[pdf > 0.1]}

            if i == 0:
                ret_dict[i]["mus_loss"] = mus_loss.unsqueeze(0)
                ret_dict[i]["sig_loss"] = sig_loss.unsqueeze(0)
                ret_dict[i]["mus_reg"] = mus_reg.unsqueeze(0)
                ret_dict[i]["sig_reg"] = sig_reg.unsqueeze(0)


            if depth_analysis_validation:
                ret_dict[i]["uniform_incell_pdf_to_plot"] = get_uniform_incell_pdf(t_vals, weights, self.cfg)
                ret_dict[i]["t_vals_for_plot"] = t_vals
                if i == 1:
                    ret_dict[i]["gaussian_incell_pdf_to_plot"] = get_gaussian_incell_pdf(t_vals_0.detach(), weights_0,
                                        mus_0, sigmas_0, part_inside_cells_0.detach(), self.cfg)

                    ret_dict[i]["smoothed_gaussian_incell_pdf_to_plot"] = get_gaussian_incell_pdf(t_vals_0.detach(), weights_0,
                                                                                         mus_0, smoothed_sigmas,
                                                                                         smoothed_part_inside_bins.detach(),
                                                                                         self.cfg)


        return ret_dict
