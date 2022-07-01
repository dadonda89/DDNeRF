from nerf.math_utils import *
import numpy as np
from typing import Optional
from copy import deepcopy
import torch
import pdb
# import torchsearchsorted

def get_combined_samples(cfg, near, far, mode):

    t_vals = torch.linspace(
        0.0,
        1.0,
        cfg.nerf[mode]['num_coarse']//2 + 1,
        dtype=near.dtype,
        device=near.device,
    )

    t_vals_uniform = cfg.dataset.near * (1.0 - t_vals) + cfg.dataset.combined_split * t_vals

    min_d = cfg.dataset.combined_split
    max_d = far[0]

    d_i = min_d*(1.0 - t_vals) + max_d * t_vals

    t_vals_nonuniform = min_d + torch.sort(1-(torch.log2(d_i-min_d+1)/torch.log2(max_d-min_d+1)))[0]*(max_d-min_d)

    t_vals = torch.cat((t_vals_uniform, t_vals_nonuniform[1:])).expand((near.shape[0], cfg.nerf[mode]['num_coarse'] + 1))

    return t_vals


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

    try:
        if cfg.dataset.combined_sampling_method:

            t_vals = get_combined_samples(cfg, near, far, mode)

    except:
        pass

    if cfg.nerf[mode]['perturb']:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat((mids, t_vals[..., -1:]), dim=-1)
        lower = torch.cat((t_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(t_vals.shape, dtype=near.dtype, device=near.device)
        t_vals = lower + (upper - lower) * t_rand
        t_vals[:, 0] = near.squeeze()
        t_vals[:, -1] = far.squeeze()

    return t_vals

def sample_pdf(bins, weights, num_samples, cfg, det=True):
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

    if cfg.train_params.pdf_padding:
        weights = weights_blur + 0.01

    else:
        prev = weights_pad[..., :-2]
        next = weights_pad[..., 2:]

        weights = 0.8*weights + 0.1*prev + 0.1*next + 0.01

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
        u = torch.minimum(u, torch.tensor(0.9999))

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


def sample_pdf_with_mu_sigma(bins, weights, mus, sigmas, part_inside_bins, left_tail, num_samples, cfg, det=True):

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

    if cfg.train_params.pdf_padding:
        weights = weights_blur + 0.01

    else:
        prev = weights_pad[..., :-2]
        next = weights_pad[..., 2:]

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
        s_test = deepcopy(s)
        u = (torch.arange(num_samples, device=weights.device) * s).expand(list(cdf.shape[:-1]) + [num_samples])
        u1_test = deepcopy(u)
        u = u + ((torch.rand(cdf.shape[0], num_samples, device=weights.device)))/(num_samples + 1e-5)
        u2_test = deepcopy(u)

        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.tensor(0.9999))
        u3_test = deepcopy(u)
        u = torch.maximum(u, torch.tensor(0))
        u4_test = deepcopy(u)


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
        z = torch.minimum(z, torch.tensor(0.999))

        if z.sum()!=z.sum():
            pdb.set_trace()

        new_mus = torch.gather(mus, index=bins_ind.type(torch.int64), dim=-1)
        new_sigmas = torch.gather(sigmas, index=bins_ind.type(torch.int64), dim=-1)

    z = approximate_inverse_cdf(z)

    #if z.sum() != z.sum():
    #    pdb.set_trace()

    t = torch.clip(z*new_sigmas+new_mus, 0, 0.99999)

    samples = bins_g0 + t * (bins_g1 - bins_g0)

    #if samples.sum() != samples.sum():
    #    pdb.set_trace()

    samples[:, -1] = cfg.dataset.far
    samples[:, 0] = cfg.dataset.near

    samples = torch.sort(samples, dim=1)[0]

    #if samples.sum() != samples.sum():
    #    pdb.set_trace()

    return torch.nn.Parameter(samples)