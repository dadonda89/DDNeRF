import torch

from .nerf_helpers_and_samplers import cumprod_exclusive


def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    mus=None,
    cfg=None
):
    # TESTED
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    mids = (depth_values[..., 1:] + depth_values[..., :-1])/2

    dists = depth_values[..., 1:] - depth_values[..., :-1]

    delta = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :3])
    # mip nerf rgb manipulation
    rgb = rgb * (1 + 2 * 0.001) - 0.001
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    #sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    density = radiance_field[..., 3] + noise
    sigma_a = torch.nn.functional.softplus(density - 1)
    alpha = 1.0 - torch.exp(-sigma_a * delta)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    #trans = torch.exp(-torch.cat([torch.zeros_like(delta[..., :1]), torch.cumsum(delta[..., :-1], axis=-1)], axis=-1))
    #weights = alpha * trans

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)

    if cfg!=None:
        if (cfg.dataset.type).lower() == "blender" or (cfg.dataset.basedir).endswith("segmented"):
            eps_mask = torch.zeros_like(weights)

            eps_mask[:,-1] += 1e-10

            weights = weights + eps_mask.detach()

            pdf = weights/(weights.sum(1).reshape(-1, 1))

        else:
            pdf = weights

    else:
        pdf = weights

    depth_map = pdf * mids
    #depth_map = weights * mids
    depth_map = depth_map.sum(dim=-1)
    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    corrected_disp_map = None
    if mus is not None:
        sections_len = depth_values[..., 1:] - depth_values[..., :-1]
        sections_mus = depth_values[..., :-1] + mus*sections_len
        corrected_depth_map = pdf * sections_mus
        corrected_depth_map = corrected_depth_map.sum(dim=1)
        corrected_disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(corrected_depth_map), corrected_depth_map / acc_map)
        depth_map = corrected_depth_map

    return rgb_map, disp_map, acc_map, weights, depth_map, corrected_disp_map, rgb