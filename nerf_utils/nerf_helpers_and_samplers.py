from nerf_utils.math_utils import *
import numpy as np
from typing import Optional

import torch
import pdb
# import torchsearchsorted


def img2mse(img_src, img_tgt):
    return torch.nn.functional.mse_loss(img_src, img_tgt)


def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (ij meas i axis set to be x (horizontal axis) and j axis set to be y (vertical axis))
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


def get_ray_bundle(
    height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor
):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    """
    # TESTED

    epsilon = 1e-5

    ii, jj = meshgrid_xy(
        torch.arange(
            width, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ).to(tform_cam2world),
        torch.arange(
            height, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ),
    )
    directions = torch.stack(
        [
            (ii - width * 0.5) / focal_length,
            -(jj - height * 0.5) / focal_length,
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)

    ray_origins[ray_origins == 0] += epsilon
    ray_directions[ray_directions == 0] += epsilon

    dx = torch.sqrt(
        torch.sum((directions[:-1, :, :] - directions[1:, :, :]) ** 2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)

    return ray_origins, ray_directions, radii

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # UNTESTED, but fairly sure.
    #NDC(normalized device coordinates).

    # Shift rays origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def gather_cdf_util(cdf, inds):
    r"""A very contrived way of mimicking a version of the tf.gather()
    call used in the original impl.
    """
    orig_inds_shape = inds.shape
    inds_flat = [inds[i].view(-1) for i in range(inds.shape[0])]
    valid_mask = [
        torch.where(ind >= cdf.shape[1], torch.zeros_like(ind), torch.ones_like(ind))
        for ind in inds_flat
    ]
    inds_flat = [
        torch.where(ind >= cdf.shape[1], (cdf.shape[1] - 1) * torch.ones_like(ind), ind)
        for ind in inds_flat
    ]
    cdf_flat = [cdf[i][ind] for i, ind in enumerate(inds_flat)]
    cdf_flat = [cdf_flat[i] * valid_mask[i] for i in range(len(cdf_flat))]
    cdf_flat = [
        cdf_chunk.reshape([1] + list(orig_inds_shape[1:])) for cdf_chunk in cdf_flat
    ]
    return torch.cat(cdf_flat, dim=0)

def sample_first_cycle(cfg, near, far, mode):

    t_vals = torch.linspace(
        0.0,
        1.0,
        cfg.nerf[mode]['num_coarse'] + 1,
        dtype=near.dtype,
        device=near.device,
    )
    if not getattr(cfg.nerf, mode).lindisp:
        t_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)


    if cfg.nerf[mode]['perturb']:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat((mids, t_vals[..., -1:]), dim=-1)
        lower = torch.cat((t_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(t_vals.shape, dtype=near.dtype, device=near.device)
        t_vals = lower + (upper - lower) * t_rand
        t_vals[:, 0] = near.squeeze()
        t_vals[:, -1] = far.squeeze()

    #t_vals = t_vals.expand((near.shape[0], cfg.nerf[mode]['num_coarse'] + 1))
    if cfg.training_methods.non_uniform_sampling.use:
        min_d = near[0]
        max_d = far[0]

        max_v = max_d - min_d

        t_vals -= min_d  # 0..max_v

        non_lin_depth = torch.log(t_vals + 1.)  # no log with custom base in pytorch, so using this
        t_vals = (non_lin_depth / math.log(far[0] + 1))*max_v

        t_vals = -1*(t_vals-max_v)

        idx = ((t_vals.shape[1]-1-torch.arange(t_vals.shape[1], device=t_vals.device)).type(torch.LongTensor)).to(t_vals.device)

        t_vals = torch.index_select(t_vals, 1, idx)

        #t_vals = near + (torch.log(t_vals-near+1)/torch.log(far-near+1))*(far-near)

    return t_vals


def sample_pdf(bins, weights, num_samples, det=False):
    # TESTED (Carefully, line-to-line).
    # But chances of bugs persist; haven't integration-tested with
    # training routines.

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / weights.sum(-1).unsqueeze(-1)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, num_samples).to(weights)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(weights)

    # Invert CDF
    inds = torch.searchsorted(
        cdf.contiguous(), u.contiguous(), right=True
    )
    below = torch.max(torch.zeros_like(inds), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), -1)
    orig_inds_shape = inds_g.shape

    cdf_g = gather_cdf_util(cdf, inds_g)
    bins_g = gather_cdf_util(bins, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def sample_pdf_2(bins, weights, num_samples, cfg, det=True):
    r"""sample_pdf function from another concurrent pytorch implementation
    by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
    """

    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ],
        axis=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    if cfg.temp.pdf_padding:
        weights = weights_blur + 0.01
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.minimum(torch.tensor(1),torch.cumsum(pdf[..., :-1], dim=-1))
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])], dim=-1
    )  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0, 1.0, steps=num_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=weights.device) * s).expand(list(cdf.shape[:-1]) + [num_samples])
        u = u + (torch.rand(cdf.shape[0], num_samples, device=weights.device)/((1/s) + 1e-5))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.tensor(1. - 1e-5))

    """
    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    """
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)[0]
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)[0]
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)

    return torch.nn.Parameter(samples)

def sample_pdf_with_mu_sigma(bins, weights, mus, sigmas, num_samples, det=True, const_sigma = False):

    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ],
        axis=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    weights = weights_blur + 0.01
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.minimum(torch.tensor(1),torch.cumsum(pdf[..., :-1], dim=-1))
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])], dim=-1
    )  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0, 0.9999, steps=num_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=weights.device) * s).expand(list(cdf.shape[:-1]) + [num_samples])
        u = u + (torch.rand(cdf.shape[0], num_samples, device=weights.device)/((1/s) + 1e-5))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.tensor(1. - 1e-5))

    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, ind = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)[0]
        return x0, x1, ind

    bins_g0, bins_g1, bins_ind = find_interval(bins)
    cdf_g0, cdf_g1, _ = find_interval(cdf)

    t = torch.normal(0, 1, size=bins_g0.shape, device=weights.device)

    i_inds = ((torch.ones(bins_ind.shape[1], dtype=torch.float64).reshape(-1, 1) @
               torch.arange(bins_ind.shape[0], dtype=torch.float64).reshape(1, -1)).T).reshape(1,-1).type(torch.int64).squeeze()
    j_inds = bins_ind.reshape(1, -1).squeeze()

    new_mus = mus[i_inds, j_inds].reshape(bins_g0.shape[0], bins_g0.shape[1])
    new_sigmas = sigmas[i_inds, j_inds].reshape(bins_g0.shape[0], bins_g0.shape[1])

    if const_sigma:
        new_sigmas = torch.ones_like(new_sigmas, device=weights.device)*const_sigma

    samples = torch.clip(torch.sort(t * new_sigmas * (bins_g1 - bins_g0) + bins_g0 + new_mus * (bins_g1 - bins_g0), 1)[0], min=bins[0][0] + 1e-4, max=bins[0][-1] - 1e-4)

    samples = torch.sort(torch.arange(samples.shape[1], device=weights.device) * 1e-6 + samples, dim=1)[0]

    return torch.nn.Parameter(samples), bins_ind, mus, bins

def sample_pdf_with_mu_sigma_v2(t_vals, weights, mus, sigmas, num_samples, det=True, const_sigma = False, num_of_bins = 3):

    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ],
        axis=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # weights = weights_blur + 0.01

    samples = torch.zeros((t_vals.shape[0], num_samples+1), device=t_vals.device)

    # grab the relevalnt bins
    weights, relevant_bins = weights.topk(num_of_bins, dim=-1, sorted=False)

    # change the order to be like in the ray order (instead of random order)
    relevant_bins_sorted, temp_idx = torch.sort(relevant_bins, dim=-1)
    weights = torch.gather(weights, dim=-1, index=temp_idx)


    # gather relevant mus and sigmas
    mus = torch.gather(mus, dim=-1, index=temp_idx)
    sigmas = torch.gather(sigmas, dim=-1, index=temp_idx)

    partitions = torch.sort(torch.cat((relevant_bins_sorted,relevant_bins_sorted+1), dim=-1), dim=-1)[0]

    # check how many samples require for the original bins setting and how many additional samples are
    ray_partitions_samples = ((partitions[:, 1:] - partitions[:, :-1]) != 0).sum(1)
    additional_samples = num_samples - ray_partitions_samples

    # calculate pdf and cdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.minimum(torch.tensor(1), torch.cumsum(pdf[..., :-1], dim=-1))
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])], dim=-1
    )  # (batchsize, len(bins))

    # check for all optional cases
    possible_samples = additional_samples.unique()

    # filter and calc samples for each

    for i in range(possible_samples.shape[0]):

        samples_number = possible_samples[i]

        relevant_rays = (additional_samples == samples_number).nonzero()

        relevant_rays_bins = relevant_bins_sorted[relevant_rays].squeeze(1)
        relevant_cdf = cdf[relevant_rays].squeeze(1)
        relevant_mus = mus[relevant_rays].squeeze(1)
        relevant_sigmas = sigmas[relevant_rays].squeeze(1)
        relevant_partitions = partitions[relevant_rays].squeeze(1)
        relevant_t_vals = t_vals[relevant_rays].squeeze(1)


        # Take uniform samples
        if det:
            u = torch.linspace(
                0.0, 0.9999, steps=samples_number, dtype=weights.dtype, device=weights.device
            )
            u = u.expand(list(relevant_cdf.shape[:-1]) + [samples_number])
        else:
            s = 1 / samples_number
            u = (torch.arange(samples_number, device=weights.device) * s).expand(list(relevant_cdf.shape[:-1]) + [samples_number])
            u = u + (torch.rand(relevant_cdf.shape[0], samples_number, device=weights.device)/((1/s) + 1e-5))
            # `u` is in [0, 1) --- it can be zero, but it can never be 1.
            u = torch.minimum(u, torch.tensor(1. - 1e-5))

        mask = u[..., None, :] >= relevant_cdf[..., :, None]

        def find_interval(x):
            # Grab the value where `mask` switches from True to False, and vice versa.
            # This approach takes advantage of the fact that `x` is sorted.
            x0, ind = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
            x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)[0]
            return x0, x1, ind

        bins_g0, bins_g1, bins_ind = find_interval(relevant_t_vals[:,:mask.shape[1]])
        cdf_g0, cdf_g1, _ = find_interval(relevant_cdf)

        t = torch.normal(0, 1, size=bins_g0.shape, device=weights.device)

        i_inds = ((torch.ones(bins_ind.shape[1], dtype=torch.float64).reshape(-1, 1) @
                   torch.arange(bins_ind.shape[0], dtype=torch.float64).reshape(1, -1)).T).reshape(1, -1).type(torch.int64).squeeze()
        j_inds = bins_ind.reshape(1, -1).squeeze()

        new_mus = relevant_mus[i_inds, j_inds].reshape(bins_g0.shape[0], bins_g0.shape[1])
        new_sigmas = relevant_sigmas[i_inds, j_inds].reshape(bins_g0.shape[0], bins_g0.shape[1])

        relevant_inds_0 = torch.gather(relevant_rays_bins, index=bins_ind, dim=-1)
        relevant_inds_1 = relevant_inds_0 + 1

        bins_g0 = torch.gather(relevant_t_vals, index=relevant_inds_0, dim=-1)
        bins_g1 = torch.gather(relevant_t_vals, index=relevant_inds_1, dim=-1)


        if const_sigma:
            new_sigmas = torch.ones_like(new_sigmas, device=weights.device)*const_sigma

        relative_loc = torch.clip(t * new_sigmas + new_mus, min=0 + 1e-4, max=1 - 1e-4)
        samples_temp = torch.sort(relative_loc * (bins_g1 - bins_g0) + bins_g0, 1)[0]


        t_vals_0 = torch.gather(relevant_t_vals, index=relevant_rays_bins, dim=-1)
        t_vals_1 = torch.gather(relevant_t_vals, index=relevant_rays_bins+1, dim=-1)
        t_vals_for_filtering = torch.sort(torch.cat((t_vals_0[:,1:] ,t_vals_1), dim=1), 1)[0]

        positions = (relevant_partitions[:, 1:] - relevant_partitions[:, :-1]) != 0

        filtered_t_vals = t_vals_for_filtering[positions == True].reshape(-1, num_samples-samples_number)

        t_vals_for_samples = torch.cat((t_vals_0[:, :1], filtered_t_vals), dim=1)

        samples_temp = torch.sort(torch.cat((samples_temp, t_vals_for_samples), dim=1), 1)[0]

        # samples = torch.sort(torch.arange(samples.shape[1], device=weights.device) * 1e-6 + samples, dim=1)[0]

        samples[relevant_rays.squeeze()] = samples_temp

    t_vals_0 = torch.gather(t_vals, index=relevant_bins_sorted, dim=-1)
    t_vals_1 = torch.gather(t_vals, index=relevant_bins_sorted + 1, dim=-1)

    return torch.nn.Parameter(samples), mus, t_vals_0, t_vals_1, relevant_bins_sorted

def sample_pdf_with_mu_sigma_v3(t_vals, weights, mus, sigmas, num_samples, det=True, const_sigma = False):

    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ],
        axis=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    #weights = weights_blur + 0.01
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.minimum(torch.tensor(1),torch.cumsum(pdf[..., :-1], dim=-1))
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])], dim=-1
    )  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0, 0.9999, steps=num_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=weights.device) * s).expand(list(cdf.shape[:-1]) + [num_samples])
        u = u + (torch.rand(cdf.shape[0], num_samples, device=weights.device)/((1/s) + 1e-5))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.tensor(1. - 1e-5))

    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, ind = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)[0]
        return x0, x1, ind

    bins_g0, bins_g1, bins_ind = find_interval(t_vals)
    cdf_g0, cdf_g1, _ = find_interval(cdf)

    t = torch.normal(0, 1, size=bins_g0.shape, device=weights.device)

    i_inds = ((torch.ones(bins_ind.shape[1], dtype=torch.float64).reshape(-1, 1) @
               torch.arange(bins_ind.shape[0], dtype=torch.float64).reshape(1, -1)).T).reshape(1,-1).type(torch.int64).squeeze()
    j_inds = bins_ind.reshape(1, -1).squeeze()

    new_mus = mus[i_inds, j_inds].reshape(bins_g0.shape[0], bins_g0.shape[1])
    new_sigmas = sigmas[i_inds, j_inds].reshape(bins_g0.shape[0], bins_g0.shape[1])

    if const_sigma:
        new_sigmas = torch.ones_like(new_sigmas, device=weights.device)*const_sigma

    samples = torch.clip(torch.sort(t * new_sigmas * (bins_g1 - bins_g0) + bins_g0 + new_mus * (bins_g1 - bins_g0), 1)[0], min=t_vals[0][0] + 1e-4, max=t_vals[0][-1] - 1e-4)

    samples = torch.sort(torch.arange(samples.shape[1], device=weights.device) * 1e-6 + samples, dim=1)[0]

    return torch.nn.Parameter(samples), pdf, mus, sigmas, t_vals

def sample_pdf_with_mu_sigma_v4(bins, weights, mus, sigmas, num_samples, det=True, const_sigma = False):

    """
    this version is samples with respect to the approximated mids of the sections
    """

    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ],
        axis=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    weights = weights_blur + 0.01
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.minimum(torch.tensor(1),torch.cumsum(pdf[..., :-1], dim=-1))
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])], dim=-1
    )  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0, 0.9999, steps=num_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=weights.device) * s).expand(list(cdf.shape[:-1]) + [num_samples])
        u = u + (torch.rand(cdf.shape[0], num_samples, device=weights.device)/((1/s) + 1e-5))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.tensor(1. - 1e-5))

    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, ind = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)[0]
        return x0, x1, ind

    bins_g0, bins_g1, bins_ind = find_interval(bins)
    cdf_g0, cdf_g1, _ = find_interval(cdf)

    t = torch.normal(0, 1, size=(bins_g0.shape[0], num_samples), device=weights.device)

    i_inds = ((torch.ones(bins_ind.shape[1], dtype=torch.float64).reshape(-1, 1) @
               torch.arange(bins_ind.shape[0], dtype=torch.float64).reshape(1, -1)).T).reshape(1,-1).type(torch.int64).squeeze()
    j_inds = bins_ind.reshape(1, -1).squeeze()

    new_mus = mus[i_inds, j_inds].reshape(bins_g0.shape[0], bins_g0.shape[1])
    new_sigmas = sigmas[i_inds, j_inds].reshape(bins_g0.shape[0], bins_g0.shape[1])

    if const_sigma:
        new_sigmas = torch.ones_like(new_sigmas, device=weights.device)*const_sigma

    mids = torch.clip(torch.sort(t * new_sigmas * (bins_g1 - bins_g0) + bins_g0 + new_mus * (bins_g1 - bins_g0), 1)[0], min=bins[0][0] + 1e-4, max=bins[0][-1] - 1e-4)

    mids = torch.sort(torch.arange(mids.shape[1], device=weights.device) * 1e-6 + mids, dim=1)[0]

    t_vals = (mids[:, 1:] + mids[:, :-1])/2

    t_start = mids[:, 0] - (t_vals[:, 0] - mids[:, 0])

    t_end = mids[:, -1] + (mids[:, -1] - t_vals[:, -1])

    t_vals = torch.clip(torch.cat((t_start.reshape(-1,1), t_vals, t_end.reshape(-1,1)), dim=1), min=bins[0][0] + 1e-4, max=bins[0][-1] - 1e-4)

    return torch.nn.Parameter(t_vals)

def sample_pdf_with_mu_sigma_v5(bins, weights, mus, sigmas, part_inside_bins, left_tail, num_samples, cfg, det=True):

    """
    this version is samples with respect to the approximated mids of the sections
    """

    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ],
        axis=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    if cfg.temp.pdf_padding:
        weights = weights_blur + 0.01

    else:
        prev = weights_pad[..., :-2]
        next = weights_pad[...,2:]

        weights = 0.8*weights + 0.1*prev + 0.1*next + 0.01

    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.minimum(torch.tensor(1), torch.cumsum(pdf[..., :-1], dim=-1))
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])], dim=-1
    )  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0, 0.9999, steps=num_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        s = 1 / (num_samples-1)
        u = (torch.arange(num_samples, device=weights.device) * s).expand(list(cdf.shape[:-1]) + [num_samples])
        u = u + ((torch.rand(cdf.shape[0], num_samples, device=weights.device))-0.5)/num_samples
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.tensor(1. - 1e-5))
        u = torch.maximum(u, torch.tensor(0))

    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, ind = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)[0]
        return x0, x1, ind

    # the case where only one cell is in the coarse ner:
    if bins.shape[1] == 2:
        z = u *part_inside_bins + left_tail
        new_mus = mus
        new_sigmas = sigmas
        bins_g0 = bins[:, 0].reshape(-1,1)
        bins_g1 = bins[:, 1].reshape(-1,1)
    else:
        bins_g0, bins_g1, bins_ind = find_interval(bins)
        cdf_g0, cdf_g1, _ = find_interval(cdf)

        part_inside_bins = torch.gather(part_inside_bins, index=bins_ind.type(torch.int64), dim=-1)
        left_tail = torch.gather(left_tail, index=bins_ind.type(torch.int64), dim=-1)

        z = (((u - cdf_g0)/(cdf_g1-cdf_g0))*part_inside_bins + left_tail)

        new_mus = torch.gather(mus, index=bins_ind.type(torch.int64), dim=-1)
        new_sigmas = torch.gather(sigmas, index=bins_ind.type(torch.int64), dim=-1)

    z = approximate_inverse_cdf(z)

    t = torch.clip(z*new_sigmas+new_mus, 0, 0.99)

    samples = bins_g0 + t * (bins_g1 - bins_g0)

    samples[:, -1] = cfg.dataset.far
    samples[:, 0] = cfg.dataset.near

    samples = torch.sort(samples, dim=1)[0]

    return torch.nn.Parameter(samples)