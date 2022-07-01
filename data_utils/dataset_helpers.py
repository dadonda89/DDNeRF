import torch

def ndc_mipnerf_rays(H, W, focal, rays_o, rays_d, near=1):
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

    mat = rays_o
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = torch.sqrt(torch.sum((mat[:-1, :, :] - mat[1:, :, :]) ** 2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]])

    dy = torch.sqrt(torch.sum((mat[:, :-1, :] - mat[ :, 1:, :]) ** 2, -1))
    dy = torch.cat([dy, dy[:, -2:-1]], 1)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = (0.5 * (dx + dy)) * 2 / torch.sqrt(torch.tensor(12))

    return rays_o, rays_d, radii


def switch_t_ndc_to_regular(ndc_depth, rays_o, rays_d):

    regular_depth = ndc_depth*rays_o[:,:,-1]/(rays_d[:,:,-1]-ndc_depth*rays_d[:,:,-1]) + 1

    return regular_depth