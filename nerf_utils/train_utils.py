import torch
import pdb
import numpy as np
from nerf.nerf_helpers_and_samplers import get_minibatches, ndc_rays, sample_first_cycle, sample_pdf_with_mu_sigma, \
    sample_pdf_with_mu_sigma_v2, sample_pdf_with_mu_sigma_v3, sample_pdf_with_mu_sigma_v4, sample_pdf_with_mu_sigma_v5
from nerf.nerf_helpers_and_samplers import sample_pdf_2 as sample_pdf

from nerf.volume_rendering_utils import volume_render_radiance_field
from nerf import get_ray_bundle, meshgrid_xy
from nerf.math_utils import *
from nerf.loss import *


def run_network(network_fn, enc_samples, ray_batch, chunksize, embeddirs_fn):

    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(viewdirs.shape[0], enc_samples.shape[1], viewdirs.shape[2])
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((enc_samples.reshape(-1, enc_samples.shape[-1]), embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    preds = [network_fn(batch.type(torch.float)) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)

    radiance_field = radiance_field.reshape(
        list(enc_samples.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field



def predict_and_render_radiance_with_mu_sigma(
    ray_batch,
    model,
    cfg,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    cycles_number=2
):
    """
    this version take all partition in the first pass to estimate second pass mu and rgb
    """

    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    rr = ray_batch[..., 6].reshape(-1, 1)
    bounds = ray_batch[..., 7:9].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    ret = []
    ret_dict = {}

    for i in range(cycles_number):

        if i == 0:
            t_vals = sample_first_cycle(cfg, near, far, mode)

        else:
            if mode=='train' or not cfg.models.val_sampling:
                samples_num = getattr(cfg.nerf, mode).num_fine + 1
                free_sampling = False
            else:
                samples_num = getattr(cfg.nerf, mode).num_fine + getattr(cfg.nerf, mode).num_coarse + 2
                free_sampling = True
            t_vals, bins_ind, mus_0, t_vals_0 = sample_pdf_with_mu_sigma(
                t_vals,
                weights,
                mus,
                sigmas,
                samples_num,
                det=(getattr(cfg.nerf, mode).perturb == 0.0),
                const_sigma=cfg.models.const_sigma
            )
        if i == 1 and not free_sampling:
            t_vals = torch.sort(torch.cat((t_vals, t_vals_0), 1), 1)[0]

        samples = cast_rays(t_vals, ro, rd, rr, cfg.nerf.ray_shape)

        enc_samples = encode_position_fn(samples, cfg.models.max_ipe_deg, cfg.models.min_ipe_deg)

        radiance_field = run_network(model, enc_samples, ray_batch, getattr(cfg.nerf, mode).chunksize, encode_direction_fn)

        raw_mus, raw_sigmas = radiance_field[:,:,-2], radiance_field[:, :, -1]

        mus = torch.sigmoid(raw_mus)
        sigmas = torch.sigmoid(raw_sigmas)

        radiance_field = radiance_field[:, :, :-2]

        rgb, disp, acc, weights, depth, corrected_disp_map, rgb_raw = volume_render_radiance_field(radiance_field, t_vals, rd,
            radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,
            white_background=getattr(cfg.nerf, mode).white_background, mus=mus
        )

        bfp_95 = bins_for_percentage(weights, 0.95)
        bfp_90 = bins_for_percentage(weights, 0.9)
        bfp_80 = bins_for_percentage(weights, 0.8)

        if i == 0:
            rgb_coarse_raw = rgb_raw

        dp_loss, raw_rgb_loss = None, None
        if i == 1:
            if not free_sampling:
                dp_loss = estimate_dp_loss(t_vals, weights, mus_0, bins_ind, t_vals_0)
                dp_loss = dp_loss.unsqueeze(0)
                raw_rgb_loss = estimate_rgb_loss(t_vals, weights, rgb_coarse_raw, rgb_raw, t_vals_0)
                raw_rgb_loss = raw_rgb_loss.unsqueeze(0)

        ret = ret + [rgb, disp, acc, weights, depth, mus, sigmas, dp_loss, corrected_disp_map, raw_rgb_loss]

        ret_dict[i] = {"rgb": rgb, "disp": disp, "acc": acc, "weights": weights, "depth": depth, "mus": mus,
                       "sigmas": sigmas, "dp_loss": dp_loss, "corrected_disp_map": corrected_disp_map,
                       "raw_rgb_loss": raw_rgb_loss, "bfp_95": bfp_95, "bfp_90": bfp_90, "bfp_80": bfp_80}

    return ret

def predict_and_render_radiance_with_mu_sigma_v2(
    ray_batch,
    model,
    cfg,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    cycles_number=2
):
    """
    v2 is the version with the top-k sections chosen
    """

    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    rr = ray_batch[..., 6].reshape(-1, 1)
    bounds = ray_batch[..., 7:9].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    ret = []

    ret_dict = {}

    for i in range(cycles_number):

        if i == 0:
            t_vals = sample_first_cycle(cfg, near, far, mode)

        else:
            t_vals, mus_0, t_vals_0, t_vals_1, relevant_bins_sorted = sample_pdf_with_mu_sigma_v2(
                t_vals,
                weights,
                mus,
                sigmas,
                getattr(cfg.nerf, mode).num_fine,
                det=(getattr(cfg.nerf, mode).perturb == 0.0),
                const_sigma=cfg.models.const_sigma,
                num_of_bins=cfg.models.top_k
            )

        samples = cast_rays(t_vals, ro, rd, rr, cfg.nerf.ray_shape)

        enc_samples = encode_position_fn(samples, cfg.models.max_ipe_deg, cfg.models.min_ipe_deg)

        radiance_field = run_network(model, enc_samples, ray_batch, getattr(cfg.nerf, mode).chunksize, encode_direction_fn)

        raw_mus, raw_sigmas = radiance_field[:, :, -2], radiance_field[:, :, -1]

        mus = torch.sigmoid(raw_mus)
        sigmas = torch.sigmoid(raw_sigmas)

        radiance_field = radiance_field[:, :, :-2]

        rgb, disp, acc, weights, depth, corrected_disp_map, rgb_raw = volume_render_radiance_field(radiance_field, t_vals, rd,
            radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,
            white_background=getattr(cfg.nerf, mode).white_background, mus=mus
        )

        bfp_95 = bins_for_percentage(weights, 0.95)
        bfp_90 = bins_for_percentage(weights, 0.9)
        bfp_80 = bins_for_percentage(weights, 0.8)

        if i == 0:
            rgb_coarse_raw = rgb_raw

        dp_loss, raw_rgb_loss = None, None
        if i == 1:
            dp_loss = estimate_dp_loss_v2(t_vals, weights, mus_0, t_vals_0, t_vals_1)
            dp_loss = dp_loss.unsqueeze(0)
            raw_rgb_loss = estimate_rgb_loss_v2(t_vals, weights, rgb_coarse_raw, rgb_raw, t_vals_0, t_vals_1, relevant_bins_sorted)
            raw_rgb_loss = raw_rgb_loss.unsqueeze(0)

        ret_dict[i] = {"rgb": rgb, "disp": disp, "acc": acc, "weights": weights, "depth": depth, "mus": mus,
                       "sigmas": sigmas, "dp_loss": dp_loss, "corrected_disp_map": corrected_disp_map,
                      "raw_rgb_loss": raw_rgb_loss, "bfp_95": bfp_95, "bfp_90": bfp_90, "bfp_80": bfp_80}

        #ret = ret + [rgb, disp, acc, weights, depth, mus, sigmas, dp_loss, corrected_disp_map, raw_rgb_loss]

    return ret_dict

def predict_and_render_radiance_with_mu_sigma_v3(
    ray_batch,
    model,
    cfg,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    cycles_number=2
):
    """
    this is the version with the cdf estimation
    """

    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    rr = ray_batch[..., 6].reshape(-1, 1)
    bounds = ray_batch[..., 7:9].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    ret_dict = {}

    for i in range(cycles_number):

        if i == 0:
            t_vals = sample_first_cycle(cfg, near, far, mode)

        else:
            t_vals, pdf_0, mus_0, sigmas_0, t_vals_0 = sample_pdf_with_mu_sigma_v3(
                t_vals,
                weights,
                mus,
                sigmas,
                getattr(cfg.nerf, mode).num_fine + 1,
                det=(getattr(cfg.nerf, mode).perturb == 0.0),
                const_sigma=cfg.models.const_sigma
            )

        samples = cast_rays(t_vals, ro, rd, rr, cfg.nerf.ray_shape)

        enc_samples = encode_position_fn(samples, cfg.models.max_ipe_deg, cfg.models.min_ipe_deg)

        radiance_field = run_network(model, enc_samples, ray_batch, getattr(cfg.nerf, mode).chunksize, encode_direction_fn)

        raw_mus, raw_sigmas = radiance_field[:, :, -2], radiance_field[:, :, -1]

        mus = torch.sigmoid(raw_mus)
        sigmas = torch.sigmoid(raw_sigmas)
        if cfg.models.const_sigma:
            sigmas = 0*sigmas + cfg.models.const_sigma

        radiance_field = radiance_field[:, :, :-2]

        rgb, disp, acc, weights, depth, corrected_disp_map, rgb_raw = volume_render_radiance_field(radiance_field, t_vals, rd,
            radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,
            white_background=getattr(cfg.nerf, mode).white_background, mus=mus
        )

        bfp_95 = bins_for_percentage(weights, 0.95)
        bfp_90 = bins_for_percentage(weights, 0.9)
        bfp_80 = bins_for_percentage(weights, 0.8)


        dp_loss = None
        if i == 1:
            pdf_1 = weights / torch.sum(weights, dim=-1, keepdim=True)
            dp_loss = estimate_dp_loss_v3(t_vals, pdf_1, pdf_0, mus_0, sigmas_0, t_vals_0)
            dp_loss = dp_loss.unsqueeze(0)


        ret_dict[i] = {"rgb": rgb, "disp": disp, "acc": acc, "weights": weights, "depth": depth, "mus": mus,
                       "sigmas": sigmas, "dp_loss": dp_loss, "corrected_disp_map": corrected_disp_map,
                      "bfp_95": bfp_95, "bfp_90": bfp_90, "bfp_80": bfp_80}


    return ret_dict

def predict_and_render_radiance_with_mu_sigma_v4(
    ray_batch,
    model,
    cfg,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    cycles_number=2
):
    """
    v4 is the version with the bins partition to calc the dp loss,
    also the samples taken as the middle of the sections
    """

    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    rr = ray_batch[..., 6].reshape(-1, 1)
    bounds = ray_batch[..., 7:9].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    ret_dict = {}

    for i in range(cycles_number):

        if i == 0:
            t_vals = sample_first_cycle(cfg, near, far, mode)

        else:
            t_vals = sample_pdf_with_mu_sigma_v4(
                t_vals,
                weights,
                mus,
                sigmas,
                getattr(cfg.nerf, mode).num_fine,
                det=(getattr(cfg.nerf, mode).perturb == 0.0),
                const_sigma=cfg.models.const_sigma
            )

        samples = cast_rays(t_vals, ro, rd, rr, cfg.nerf.ray_shape)

        enc_samples = encode_position_fn(samples, cfg.models.max_ipe_deg, cfg.models.min_ipe_deg)

        radiance_field = run_network(model, enc_samples, ray_batch, getattr(cfg.nerf, mode).chunksize, encode_direction_fn)

        raw_mus, raw_sigmas = radiance_field[:, :, -2], radiance_field[:, :, -1]

        mus = torch.sigmoid(raw_mus)
        sigmas = torch.sigmoid(raw_sigmas)

        radiance_field = radiance_field[:, :, :-2]

        rgb, disp, acc, weights, depth, corrected_disp_map, rgb_raw = volume_render_radiance_field(radiance_field, t_vals, rd,
            radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,
            white_background=getattr(cfg.nerf, mode).white_background, mus=mus
        )

        bfp_95 = bins_for_percentage(weights, 0.95)
        bfp_90 = bins_for_percentage(weights, 0.9)
        bfp_80 = bins_for_percentage(weights, 0.8)

        if i == 0:
            rgb_coarse_raw = rgb_raw
            t_vals_0 = t_vals
            mus_0 = mus


        dp_loss, raw_rgb_loss = None, None
        if i == 1:
            dp_loss = estimate_dp_loss_v4(t_vals, weights, mus_0, t_vals_0)
            dp_loss = dp_loss.unsqueeze(0)
            #raw_rgb_loss = estimate_rgb_loss_v2(t_vals, weights, rgb_coarse_raw, rgb_raw, t_vals_0, t_vals_1, relevant_bins_sorted)
            #raw_rgb_loss = raw_rgb_loss.unsqueeze(0)

        ret_dict[i] = {"rgb": rgb, "disp": disp, "acc": acc, "weights": weights, "depth": depth, "mus": mus,
                       "sigmas": sigmas, "dp_loss": dp_loss, "corrected_disp_map": corrected_disp_map,
                      "bfp_95": bfp_95, "bfp_90": bfp_90, "bfp_80": bfp_80}

        #ret = ret + [rgb, disp, acc, weights, depth, mus, sigmas, dp_loss, corrected_disp_map, raw_rgb_loss]

    return ret_dict

def predict_and_render_radiance_with_mu_sigma_v5(
    ray_batch,
    model_1,
    model_2,
    cfg,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    cycles_number=2,
    depth_analysis_mode = False
):
    """
    v5 samples by cdf capacity and normlized distribusions of mu and sigmas to sum to one at each cell
    """

    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    rr = ray_batch[..., 6].reshape(-1, 1)
    bounds = ray_batch[..., 7:9].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    ret_dict = {}

    for i in range(cycles_number):

        if i == 0:
            model = model_1
        if i==1 and model_2:
            model = model_2
            mus = None

        if i == 0:
            t_vals = sample_first_cycle(cfg, near, far, mode)

        else:
            t_vals = sample_pdf_with_mu_sigma_v5(
                t_vals,
                weights,
                mus_0,
                sigmas_0,
                part_inside_bins,
                left_tail,
                getattr(cfg.nerf, mode).num_fine + 1,
                cfg,
                det=(getattr(cfg.nerf, mode).perturb == 0.0)
            )

        samples = cast_rays(t_vals, ro, rd, rr, cfg.nerf.ray_shape)

        enc_samples = encode_position_fn(samples, cfg.models.max_ipe_deg, cfg.models.min_ipe_deg)

        radiance_field = run_network(model, enc_samples, ray_batch, getattr(cfg.nerf, mode).chunksize, encode_direction_fn)

        if i == 0:
            raw_mus, raw_sigmas = radiance_field[:, :, -2], radiance_field[:, :, -1]

            mus_reg = cfg.temp.mu_regularization*torch.nn.functional.mse_loss(raw_mus, torch.zeros_like(raw_mus))
            sig_reg = cfg.temp.mu_regularization*torch.nn.functional.mse_loss(raw_sigmas, torch.zeros_like(raw_sigmas))

            mus = torch.sigmoid(raw_mus/cfg.temp.reduction_factor)
            sigmas = torch.sigmoid(raw_sigmas/cfg.temp.reduction_factor) + 0.001 if not cfg.models.const_sigma else torch.ones_like(mus, device=mus.device)*cfg.models.const_sigma

            if mus.sum() != mus.sum():
                pdb.set_trace()

            if sigmas.sum() != sigmas.sum():
                pdb.set_trace()

            x_0 = (0-mus)/sigmas
            x_1 = (1-mus)/sigmas

            left_tail = approximate_cdf(x_0)
            part_inside_bins = (approximate_cdf(x_1) - left_tail)

            if left_tail.sum() != left_tail.sum():
                pdb.set_trace()

            if part_inside_bins.sum() != part_inside_bins.sum():
                pdb.set_trace()

            radiance_field = radiance_field[:, :, :-2]

        rgb, disp, acc, weights, depth, corrected_disp_map, rgb_raw = volume_render_radiance_field(radiance_field, t_vals, rd,
            radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,
            white_background=getattr(cfg.nerf, mode).white_background, mus=mus, cfg=cfg
        )

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


        dp_loss, raw_rgb_loss = None, None
        if i == 1:
            #dp_loss = estimate_dp_loss_v4(t_vals, weights, mus_0, t_vals_0)
            dp_loss = estimate_dp_loss_v5(t_vals.detach(), t_vals_0.detach(), weights.detach(), weights_0, mus_0, sigmas_0, left_tails_0, part_inside_cells_0.detach(), cfg)
            dp_loss = (dp_loss + mus_reg + sig_reg).unsqueeze(0)
            #raw_rgb_loss = estimate_rgb_loss_v2(t_vals, weights, rgb_coarse_raw, rgb_raw, t_vals_0, t_vals_1, relevant_bins_sorted)
            #raw_rgb_loss = raw_rgb_loss.unsqueeze(0)

            if dp_loss.sum() != dp_loss.sum():
                pdb.set_trace()


        if mus is not None:
            pdf = (weights / torch.sum(weights, dim=-1, keepdim=True))
            mus_to_record = mus[pdf > 0.1]
            sigmas_to_record = sigmas[pdf > 0.1]

        ret_dict[i] = {"rgb": rgb, "disp": disp, "acc": acc, "weights": weights, "depth": depth, "mus": mus_to_record,
                       "sigmas": sigmas_to_record, "dp_loss": dp_loss, "corrected_disp_map": corrected_disp_map,
                      "bfp_95": bfp_95, "bfp_90": bfp_90, "bfp_80": bfp_80, "weights_sum": weights.sum(-1)}

        if depth_analysis_mode:
            ret_dict[i]["uniform_incell_pdf_to_plot"] = get_uniform_incell_pdf(t_vals, weights, cfg)

            if i == 1:
                ret_dict[i]["gaussian_incell_pdf_to_plot"] = get_gaussian_incell_pdf(t_vals_0.detach(), weights_0, mus_0, sigmas_0,
                                                      part_inside_cells_0.detach(), cfg)

                ret_dict[i]["t_vals_for_plot"] = t_vals

    return ret_dict


def predict_and_render_radiance(
    ray_batch,
    model,
    model_2,
    cfg,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    cycles_number=2,
    depth_analysis_mode=False
):
    """
    this one is for the original version (no mu and sigma)
    """

    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    rr = ray_batch[..., 6].reshape(-1, 1)
    bounds = ray_batch[..., 7:9].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    ret_dict = {}

    for i in range(cycles_number):

        if i == 0:
            t_vals = sample_first_cycle(cfg, near, far, mode)

        else:
            t_vals = sample_pdf(
                t_vals,
                weights,
                getattr(cfg.nerf, mode).num_fine + 1,
                cfg,
                det=(getattr(cfg.nerf, mode).perturb == 0.0),
            )

            t_vals = t_vals.detach()

        samples = cast_rays(t_vals, ro, rd, rr, cfg.nerf.ray_shape)

        enc_samples = encode_position_fn(samples, cfg.models.max_ipe_deg, cfg.models.min_ipe_deg)

        radiance_field = run_network(model, enc_samples, ray_batch, getattr(cfg.nerf, mode).chunksize, encode_direction_fn)

        rgb, disp, acc, weights, depth, _, _ = volume_render_radiance_field(radiance_field, t_vals, rd,
            radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,
            white_background=getattr(cfg.nerf, mode).white_background
        )

        bfp_95 = bins_for_percentage(weights, 0.95)
        bfp_90 = bins_for_percentage(weights, 0.9)
        bfp_80 = bins_for_percentage(weights, 0.8)

        ret_dict[i] = {"rgb": rgb, "disp": disp, "acc": acc, "weights": weights, "depth": depth,
                       "bfp_95": bfp_95, "bfp_90": bfp_90, "bfp_80": bfp_80}

        if depth_analysis_mode:
            ret_dict[i]["uniform_incell_pdf_to_plot"] = get_uniform_incell_pdf(t_vals, weights, cfg)

            if i==1:
                ret_dict[i]["t_vals_for_plot"] = t_vals

    return ret_dict


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    model,
    ray_origins,
    ray_directions,
    ray_rad,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    ray_depth=None
):
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    restore_shapes += restore_shapes
    restore_shapes.append(ray_directions.shape[:-1])


    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, options.dataset.ndc_coef, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.reshape((-1, 3))
        ray_rad = ray_rad.reshape((-1, 1))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])  # * (1 + options.nerf.train.depth.buffer)

    max_depth = None
    if ray_depth != None:
        max_depth = ray_depth.resize(ray_depth.shape[0], 1)

    rays = torch.cat((ro, rd, ray_rad, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = [
        predict_and_render_radiance(
            batch,
            model,
            options,
            mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn

        )
        for batch in batches
    ]

    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images[:-2]
    ]

    coarse_depth_loss = 0
    for i in range(len(pred)):
        coarse_depth_loss += pred[i][7]

    fine_depth_loss = 0
    for i in range(len(pred)):
        fine_depth_loss += pred[i][8]

    losses = [coarse_depth_loss, fine_depth_loss]

    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).

        return tuple(synthesized_images)

    return tuple(synthesized_images + losses)

def run_one_iter_of_mipnerf(
    height,
    width,
    focal_length,
    model_1,
    model_2,
    ray_origins,
    ray_directions,
    ray_rad,
    cfg,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    cycles_number=2,
    depth_analysis_mode=False
):
    predict_func = predict_and_render_radiance
    if cfg.models.type == 'DepthMipNeRFModel':
        predict_func = predict_and_render_radiance_with_mu_sigma_v5

    viewdirs = None
    if cfg.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
        None,
        ray_directions.shape[:-1]
    ]

    restore_shapes_dict = {}
    restore_shapes_dict['rgb'] = ray_directions.shape
    restore_shapes_dict['depth'] = ray_directions.shape[:-1]

    if cfg.models.type == 'DepthMipNeRFModel':
        restore_shapes = restore_shapes + [None, None, None, ray_directions.shape[:-1], None]

    if cycles_number == 2:
        restore_shapes += restore_shapes

    if cfg.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, cfg.dataset.ndc_coef, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.reshape((-1, 3))
        ray_rad = ray_rad.reshape((-1, 1))

    near = cfg.dataset.near * torch.ones_like(rd[..., :1])
    far = cfg.dataset.far * torch.ones_like(rd[..., :1])

    rays = torch.cat((ro, rd, ray_rad, near, far), dim=-1)
    if cfg.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(cfg.nerf, mode).chunksize)
    pred = [
        predict_func(
            batch,
            model_1,
            model_2,
            cfg,
            mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            cycles_number=cycles_number,
            depth_analysis_mode=depth_analysis_mode
        )
        for batch in batches
    ]

    output = pred[0]

    for i in range(1,len(pred)):
        for j in range(len(output)):
            for key in pred[i][j].keys():
                if (pred[i][j][key] is not None) and (pred[i][j][key] is not False):
                    output[j][key] = torch.cat((output[j][key], pred[i][j][key]), dim=0)


    # reshape for validation
    if mode == "validation" and not depth_analysis_mode:
        """
        synthesized_images = [
            image.view(shape) if shape is not None else image
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]
        """
        for i in range(len(output)):
            output[i]["rgb"] = output[i]["rgb"].view(restore_shapes_dict['rgb'])
            output[i]["disp"] = output[i]["disp"].view(restore_shapes_dict['depth'])
            output[i]["acc"] = output[i]["acc"].view(restore_shapes_dict['depth'])
            output[i]["depth"] = output[i]["depth"].view(restore_shapes_dict['depth'])
            if ("corrected_disp_map" in output[i].keys()) and (output[i]["corrected_disp_map"] is not None):
                output[i]["corrected_disp_map"] = output[i]["corrected_disp_map"].view(restore_shapes_dict['depth'])
    return output


def get_rays_and_target_for_iter(img_target, pose_target, H, W, focal, device, cfg):

    ray_origins, ray_directions, radii = get_ray_bundle(H, W, focal, pose_target)

    regular_rays_number = cfg.nerf.train.num_random_rays
    """

    if cfg.training_methods.segmentation_is.use:
        segmented_rays_number = int(cfg.training_methods.segmentation_is.part * cfg.nerf.train.num_random_rays)
        regular_rays_number = int(regular_rays_number - segmented_rays_number)
    """
    coords = torch.stack(
        meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
        dim=-1,
    )
    coords = coords.reshape((-1, 2))  # tensor with shape [HxW, 2] to sample ray pixels from
    select_inds = np.random.choice(
        coords.shape[0], size=regular_rays_number, replace=False
    )
    select_coords = coords[select_inds]

    """
    if cfg.training_methods.segmentation_is.use:
        seg_inds = torch.where((masks[img_idx] > 0))
        seg_coords = torch.stack((seg_inds[0], seg_inds[1]), dim=-1).reshape((-1, 2)).to(device)
        select_inds_seg = np.random.choice(seg_coords.shape[0], size=segmented_rays_number, replace=False)
        select_coords_seg = seg_coords[select_inds_seg]

        select_coords = torch.cat((select_coords, select_coords_seg))
    """

    ray_origins = ray_origins[select_coords[:, 0], select_coords[:, 1], :]
    ray_directions = ray_directions[select_coords[:, 0], select_coords[:, 1], :]
    ray_rad = radii[select_coords[:, 0], select_coords[:, 1], :].reshape(-1, 1)
    target_s = img_target[select_coords[:, 0], select_coords[:, 1], :]

    return ray_origins, ray_directions, ray_rad, target_s


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.

  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.

  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.

  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  t = np.clip(step / max_steps, 0, 1)
  log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
  return delay_rate * log_lerp





"""
ro = torch.tensor([0.15197891,0.02967256,0.00654884]).unsqueeze(0)
rd = torch.tensor([-0.37557697,  0.0464491,  -0.9842893 ]).unsqueeze(0)
rr = torch.tensor([0.00070829]).unsqueeze(0)
z_samples = torch.tensor([0. ,   0.875, 1.75 , 2.625, 3.5,   4.375, 5.25,  6.125, 7.   ]).unsqueeze(0)
samples = cast_rays(z_samples, ro, rd, rr, 'cylinder')

enc_samples = integrated_pos_enc(samples, 16, 0)
print("finnish")

"""
