import torch
from general_utils.math_utils import approximate_cdf



def estimate_dp_loss(t_vals_1, t_vals_0, pdf_1, pdf_0, mus_0, sigmas_0, left_tails_0, part_inside_cells_0, cfg=None):

    """
    this version estimate loss with sum of cdf with only one cdf per section
    """

    if (cfg.dataset.type).lower() == "blender":

        # collect only rays that intersect with dense areas

        relevent_rows = pdf_1.sum(1) > 1e-10

        # if all rays with no depth..
        if relevent_rows.sum() == 0:
            return relevent_rows.sum().detach()

        pdf_0 = pdf_0[relevent_rows]
        pdf_1 = pdf_1[relevent_rows]
        mus_0 = mus_0[relevent_rows]
        sigmas_0 = sigmas_0[relevent_rows]
        part_inside_cells_0 = part_inside_cells_0[relevent_rows]
        t_vals_1 = t_vals_1[relevent_rows]
        t_vals_0 = t_vals_0[relevent_rows]

    epsilon = 1e-12
    pdf_0 = (pdf_0 + epsilon)/torch.sum(pdf_0 + epsilon, dim=-1, keepdim=True)
    pdf_1 = (pdf_1 + epsilon)/torch.sum(pdf_1 + epsilon, dim=-1, keepdim=True)

    # transform mu, sigma from section space to ray space
    mus_0_ray = t_vals_0[:, :-1] + mus_0*(t_vals_0[:, 1:]-t_vals_0[:, :-1])
    sigmas_0_ray = sigmas_0*(t_vals_0[:, 1:]-t_vals_0[:, :-1])

    cdf = torch.minimum(torch.tensor(1, device=pdf_0.device), torch.cumsum(pdf_0[..., :-1], dim=-1))
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])], dim=-1
    )  # (batchsize, len(bins))

    mask = t_vals_1[..., None, :] > t_vals_0[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, ind = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)[0]
        return x0, x1, ind

    est_cdf, _, idx = find_interval(cdf)

    mus = torch.gather(mus_0_ray, index=idx.type(torch.int64), dim=-1)
    sigmas = torch.gather(sigmas_0_ray, index=idx.type(torch.int64), dim=-1)
    part_inside_cells = torch.gather(part_inside_cells_0, index=idx.type(torch.int64), dim=-1)
    left_tails = torch.gather(left_tails_0, index=idx.type(torch.int64), dim=-1)
    pdf = torch.gather(pdf_0, index=idx.type(torch.int64), dim=-1)

    x = (t_vals_1 - mus) / sigmas

    additional_probs = ((approximate_cdf(x) - left_tails)/part_inside_cells) * pdf

    est_cdf += additional_probs

    est_cdf[est_cdf > 1] = 1

    estimated_pdf_1 = est_cdf[:, 1:] - est_cdf[:, :-1]

    estimated_pdf_1[estimated_pdf_1 < 0] = 0

    estimated_pdf_1 = (estimated_pdf_1 + epsilon)/torch.sum(estimated_pdf_1 + epsilon, dim=-1, keepdim=True)

    # this try except is for supporting both config versions

    loss = torch.nn.functional.kl_div(estimated_pdf_1.log(), pdf_1.detach())

    return loss
