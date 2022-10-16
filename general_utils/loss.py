import torch
import numpy as np
from general_utils.math_utils import approximate_cdf
import pdb


def estimate_dp_loss(t_vals, weights, mus_0, bins_ind, t_vals_0):

    est_mus = torch.zeros(mus_0.shape, device=mus_0.device)

    weights = weights + 1e-6

    mids = 0.5*(t_vals[:, :-1] + t_vals[:, 1:])

    for i in range(est_mus.shape[-1]):

        lower = t_vals >= t_vals_0[:, i].reshape(-1, 1)
        upper = t_vals < t_vals_0[:, i+1].reshape(-1, 1)
        relevant = lower*upper

        relevant_weights = weights*relevant[:, :-1]

        denom = (relevant_weights.sum(dim=1)).reshape(-1, 1)

        relevant_weight_normalize = relevant_weights/denom

        est_mus[:, i] = ((relevant_weight_normalize*mids).sum(dim=1)-t_vals_0[:, i])/(t_vals_0[:, i+1]-t_vals_0[:, i])

    dp_loss = torch.nn.functional.mse_loss(est_mus.detach(), mus_0)

    return dp_loss

def estimate_dp_loss_v2(t_vals, weights, mus_0, t_vals_0, t_vals_1):

    """
    this is the version that use top-k bins
    """

    est_mus = torch.zeros(mus_0.shape, device=mus_0.device)

    weights = weights + 1e-6

    mids = 0.5*(t_vals[:, :-1] + t_vals[:, 1:])

    for i in range(t_vals_0.shape[-1]):

        lower = t_vals >= t_vals_0[:, i].reshape(-1, 1)
        upper = t_vals < t_vals_1[:, i].reshape(-1, 1)
        relevant = lower*upper

        relevant_weights = weights*relevant[:, :-1]

        denom = (relevant_weights.sum(dim=1)).reshape(-1, 1)

        relevant_weight_normalize = relevant_weights/denom

        est_mus[:, i] = ((relevant_weight_normalize*mids).sum(dim=1)-t_vals_0[:, i])/(t_vals_1[:, i]-t_vals_0[:, i])

    dp_loss = torch.nn.functional.mse_loss(est_mus.detach(), mus_0)

    return dp_loss

def estimate_rgb_loss(t_vals, weights, rgb_coarse, rgb_fine, t_vals_0):

    weights = weights + 1e-6

    est_rgb = torch.zeros(rgb_coarse.shape, device=rgb_coarse.device)

    for i in range(est_rgb.shape[1]):
        lower = t_vals >= t_vals_0[:, i].reshape(-1, 1)
        upper = t_vals < t_vals_0[:, i + 1].reshape(-1, 1)
        relevant = lower * upper

        relevant_weights = weights * relevant[:, :-1]

        denom = (relevant_weights.sum(dim=1)).reshape(-1, 1)

        relevant_weight_normalize = relevant_weights / denom

        est_rgb[:, i] = (relevant_weight_normalize[..., None] * rgb_fine).sum(dim=1)

    rgb_coarse_loss = torch.nn.functional.mse_loss(est_rgb.detach(), rgb_coarse)

    return rgb_coarse_loss

def estimate_rgb_loss_v2(t_vals, weights, rgb_coarse, rgb_fine, t_vals_0, t_vals_1, bins_ind):

    """
    this is the version that use top-k bins
    """

    weights = weights + 1e-6

    r = torch.gather(rgb_coarse[:,:,0], index=bins_ind, dim=1).unsqueeze(-1)
    g = torch.gather(rgb_coarse[:, :, 1], index=bins_ind, dim=1).unsqueeze(-1)
    b = torch.gather(rgb_coarse[:, :, 2], index=bins_ind, dim=1).unsqueeze(-1)

    rgb_coarse = torch.cat((r,g,b), dim=-1)

    est_rgb = torch.zeros(rgb_coarse.shape, device=rgb_coarse.device)

    for i in range(est_rgb.shape[1]):
        lower = t_vals >= t_vals_0[:, i].reshape(-1, 1)
        upper = t_vals < t_vals_1[:, i].reshape(-1, 1)
        relevant = lower * upper

        relevant_weights = weights * relevant[:, :-1]

        denom = (relevant_weights.sum(dim=1)).reshape(-1, 1)

        relevant_weight_normalize = relevant_weights / denom

        est_rgb[:, i] = (relevant_weight_normalize[..., None] * rgb_fine).sum(dim=1)

    rgb_coarse_loss = torch.nn.functional.mse_loss(est_rgb.detach(), rgb_coarse)

    return rgb_coarse_loss


def estimate_dp_loss_v3(t_vals, pdf_1, pdf_0, mus_0, sigmas_0, t_vals_0):

    """
    this version estimate loss with sum of cdf
    """

    epsilon = 1e-4

    mus_0 = t_vals_0[:, :-1] + mus_0*(t_vals_0[:, 1:]-t_vals_0[:, :-1])
    sigmas_0 = sigmas_0*(t_vals_0[:, 1:]-t_vals_0[:, :-1])

    estmated_pdf_1 = torch.zeros_like(pdf_1)

    for i in range(pdf_1.shape[1]):
        t0 = t_vals[:, i].reshape(-1, 1).detach()
        t1 = t_vals[:, i+1].reshape(-1, 1).detach()

        x0 = (t0 - mus_0)/sigmas_0
        x1 = (t1 - mus_0)/sigmas_0

        cdf0 = approximate_cdf(x0)
        cdf1 = approximate_cdf(x1)

        estmated_pdf_1[:, i] = (pdf_0*(cdf1-cdf0)).sum(1)

    estmated_pdf_11 = estmated_pdf_1/torch.sum(estmated_pdf_1 + epsilon, dim=-1, keepdim=True) + epsilon

    #loss = torch.nn.functional.kl_div(estmated_pdf_11.log(), pdf_1.detach() + epsilon)

    loss = torch.nn.functional.mse_loss(estmated_pdf_11, pdf_1.detach())

    if loss != loss:
        print('na!')

    return loss

def estimate_dp_loss_v4(t_vals, weights, mus_0, t_vals_0):

    """
    this is the version that use top-k bins
    """

    est_mus = torch.zeros(mus_0.shape, device=mus_0.device)

    weights = weights + 1e-6

    mids = 0.5*(t_vals[:, :-1] + t_vals[:, 1:])

    for i in range(t_vals_0.shape[-1]-1):

        lower = t_vals >= t_vals_0[:, i].reshape(-1, 1)
        need_to_be_added_before = lower[:, 0] == False
        upper = t_vals < t_vals_0[:, i+1].reshape(-1, 1)
        need_to_be_added_after = upper[:, -1] == False
        relevant = lower*upper

        false_col = torch.zeros((relevant.shape[0],1), device=relevant.device) == 1

        first_t_loc = (relevant[:, :-1] == False) * (relevant[:, 1:] == True)
        first_t_loc = torch.cat((first_t_loc, false_col), dim=1)

        last_t_loc = (relevant[:, :-1] == True) * (relevant[:, 1:] == False)
        last_t_loc = torch.cat((false_col, last_t_loc), dim=1)

        relevant_weights = weights*(relevant[:, :-1] + first_t_loc[:, :-1])

        # correct weights for bins cross the end of the partition:
        end_weights_ind_to_be_corrected = torch.where(last_t_loc)

        end_weights_ind_to_be_corrected = (end_weights_ind_to_be_corrected[0], end_weights_ind_to_be_corrected[1]-1)

        if len(end_weights_ind_to_be_corrected[0]):
            t_0_bin = t_vals[end_weights_ind_to_be_corrected]
            t_1_bin = t_vals[(end_weights_ind_to_be_corrected[0], end_weights_ind_to_be_corrected[1]+1)]
            t_1_partition = t_vals_0[:, i+1][end_weights_ind_to_be_corrected[0]]
            correction_coef_end = (t_1_partition-t_0_bin)/(t_1_bin-t_0_bin)
            #correction_coef_end[correction_coef_end < 0.5] = 0
            relevant_weights[end_weights_ind_to_be_corrected] = correction_coef_end*relevant_weights[end_weights_ind_to_be_corrected]

        # correct weights for bins cross the start of the partition:
        start_weights_ind_to_be_corrected = torch.where(first_t_loc)

        if len(start_weights_ind_to_be_corrected[0]):
            t_0_bin = t_vals[start_weights_ind_to_be_corrected]
            t_1_bin = t_vals[(start_weights_ind_to_be_corrected[0], start_weights_ind_to_be_corrected[1] + 1)]
            t_0_partition = t_vals_0[:, i][start_weights_ind_to_be_corrected[0]]
            correction_coef_start = (t_1_bin - t_0_partition) / (t_1_bin - t_0_bin)
            #correction_coef_start[correction_coef_start < 0.5] = 0
            relevant_weights[start_weights_ind_to_be_corrected] = correction_coef_start * relevant_weights[start_weights_ind_to_be_corrected]

        denom = (relevant_weights.sum(dim=1)).reshape(-1, 1) + 1e-6

        relevant_weight_normalize = relevant_weights/denom

        estimated_rows = torch.where(relevant.sum(1) > 0)[0]

        clipped_mids = torch.minimum(mids, t_vals_0[:, i+1].reshape(-1,1))
        clipped_mids = torch.maximum(clipped_mids, t_vals_0[:, i].reshape(-1,1))

        est_mus[estimated_rows, i] = ((relevant_weight_normalize*clipped_mids).sum(dim=1)[estimated_rows]-t_vals_0[estimated_rows, i])/(t_vals_0[estimated_rows, i+1]-t_vals_0[estimated_rows, i])

    relevant_estimations = (est_mus>0) & (est_mus<1)

    dp_loss = torch.nn.functional.mse_loss(est_mus[relevant_estimations].detach(), mus_0[relevant_estimations])

    return dp_loss

def estimate_dp_loss_v5(t_vals_1, t_vals_0, pdf_1, pdf_0, mus_0, sigmas_0, left_tails_0, part_inside_cells_0, cfg=None):

    """
    this version estimate loss with sum of cdf with only one cdf per section
    """

    import pdb
    if t_vals_1.sum() != t_vals_1.sum():
        pdb.set_trace()

    # filter rays if synthetic data is in use, to avoid "no depth" rays
    try:
        if (cfg.dataset.type).lower() == "blender":
            relevent_rows = pdf_1.sum(1) > 1e-10

            # if all rays with no depth..
            if relevent_rows.sum() == 0:
                return relevent_rows.sum()


            pdf_0 = pdf_0[relevent_rows]
            pdf_1 = pdf_1[relevent_rows]
            mus_0 = mus_0[relevent_rows]
            sigmas_0 = sigmas_0[relevent_rows]
            part_inside_cells_0 = part_inside_cells_0[relevent_rows]
            t_vals_1 = t_vals_1[relevent_rows]
            t_vals_0 = t_vals_0[relevent_rows]

    except:
        pass

    epsilon = 1e-12
    pdf_0 = (pdf_0 + epsilon)/torch.sum(pdf_0 + epsilon, dim=-1, keepdim=True)
    pdf_1 = (pdf_1 + epsilon)/torch.sum(pdf_1 + epsilon, dim=-1, keepdim=True)

    # transform mu, sigma from section space to ray space
    mus_0_ray = t_vals_0[:, :-1] + mus_0*(t_vals_0[:, 1:]-t_vals_0[:, :-1])
    sigmas_0_ray = sigmas_0*(t_vals_0[:, 1:]-t_vals_0[:, :-1])

    estmated_pdf_1 = torch.zeros_like(pdf_1)

    for i in range(pdf_1.shape[1]):

        start = t_vals_1[:, i].reshape(-1, 1)
        end = t_vals_1[:, i + 1].reshape(-1, 1)

        row_idx, start_cell = torch.where((start >= t_vals_0[:, :-1])*(start < t_vals_0[:, 1:]))

        end_cell = torch.where((end > t_vals_0[:, :-1])*(end <= t_vals_0[:, 1:]))[1]

        relevant_cells = torch.zeros((mus_0_ray.shape[0], mus_0.shape[1]+2), device=mus_0_ray.device).type(torch.bool)

        relevant_cells[:, 1:-1] = (end >= t_vals_0[:, :-1])*(start < t_vals_0[:, 1:])

        contained_cells = relevant_cells[:, 2:]*relevant_cells[:, :-2]

        # the case where pdf1 cell contains in one pdf0 cell:

        # grab relevant cells
        if (end_cell<start_cell).sum() > 0:
            pdb.set_trace()

        relevant_rows = row_idx[start_cell == end_cell]
        if len(relevant_rows > 0):

            relevant_cols = start_cell[relevant_rows]
            relevant_mus = mus_0_ray[(relevant_rows, relevant_cols)].reshape(-1,1)
            relevant_sigmas = sigmas_0_ray[(relevant_rows, relevant_cols)].reshape(-1,1)

            if (relevant_sigmas == 0).sum() > 0:
                pdb.set_trace()

            # transform to standard normal distribution
            x0 = (start[relevant_rows] - relevant_mus)/relevant_sigmas
            x1 = (end[relevant_rows] - relevant_mus)/relevant_sigmas

            if x1.sum() != x1.sum():
                pdb.set_trace()
                print(1)

            # estimate pdf1 for this cell
            cdf0 = approximate_cdf(x0)
            cdf1 = approximate_cdf(x1)

            estmated_pdf_1[relevant_rows, i] = (pdf_0[relevant_rows, relevant_cols].reshape(-1, 1)*(1/part_inside_cells_0[relevant_rows, relevant_cols]).reshape(-1,1)*(cdf1-cdf0)).squeeze()

        # the case where pdf1 cell contains in more than one pdf0 cell:

        # grab relevant cells
        relevant_rows = row_idx[start_cell != end_cell]
        if len(relevant_rows > 0):
            try:
                relevant_start_cols = start_cell[relevant_rows]
            except:
                import pdb
                pdb.set_trace()

            relevant_end_cols = end_cell[relevant_rows]
            relevant_start_mus = mus_0_ray[(relevant_rows, relevant_start_cols)].reshape(-1, 1)

            relevant_end_mus = mus_0_ray[(relevant_rows, relevant_end_cols)].reshape(-1, 1)
            relevant_start_sigmas = sigmas_0_ray[(relevant_rows, relevant_start_cols)].reshape(-1, 1)

            relevant_end_sigmas = sigmas_0_ray[(relevant_rows, relevant_end_cols)].reshape(-1, 1)

            # estimate the first part of the pdf for this cell
            start_cell_end_of_the_bin = t_vals_0[relevant_rows, relevant_start_cols+1].reshape(-1, 1)

            start_cell_x0 = (start[relevant_rows] - relevant_start_mus)/relevant_start_sigmas

            start_cell_x1 = (start_cell_end_of_the_bin - relevant_start_mus)/relevant_start_sigmas

            if (relevant_start_sigmas == 0).sum() > 0:
                print(1)

            try:
                if start_cell_x1.sum() != start_cell_x1.sum():
                    print(1)
            except:
                print(1)

            try:
                cdf0_start = approximate_cdf(start_cell_x0)
                cdf1_start = approximate_cdf(start_cell_x1)
            except:
                print(1)

            estmated_pdf_1[relevant_rows, i] += (pdf_0[relevant_rows, relevant_start_cols].reshape(-1, 1) *
                                                 (1/part_inside_cells_0[relevant_rows, relevant_start_cols]).reshape(-1, 1) * (cdf1_start - cdf0_start)).squeeze()

            # estimate the first part of the pdf for this cell
            end_cell_start_of_the_bin = t_vals_0[relevant_rows, relevant_end_cols].reshape(-1, 1)

            end_cell_x0 = (end_cell_start_of_the_bin-relevant_end_mus)/relevant_end_sigmas
            end_cell_x1 = (end[relevant_rows] - relevant_end_mus) / relevant_end_sigmas

            if end_cell_x1.sum() != end_cell_x1.sum():
                print(1)

            try:
                cdf0_end = approximate_cdf(end_cell_x0)
                cdf1_end = approximate_cdf(end_cell_x1)
            except:
                import pdb
                pdb.set_trace()

            estmated_pdf_1[relevant_rows, i] += (pdf_0[relevant_rows, relevant_end_cols].reshape(-1, 1) *
                                                 (1/part_inside_cells_0[relevant_rows, relevant_end_cols]).reshape(-1, 1) * (
                                                             cdf1_end - cdf0_end)).squeeze()

            # the case where there are contained cells:
            estmated_pdf_1[:, i] += (contained_cells*pdf_0).sum(1)


    estmated_pdf_1 = (estmated_pdf_1 + epsilon)/torch.sum(estmated_pdf_1 + epsilon, dim=-1, keepdim=True)

    # this try except is for supporting both config versions

    loss = torch.nn.functional.kl_div(estmated_pdf_1.log(), pdf_1.detach())

    if cfg.train_params.mse:
        m = (estmated_pdf_1 + pdf_1)/2

    if loss != loss:
        import pdb
        pdb.set_trace()
        print(1)
    return loss  #, estmated_pdf_1

def estimate_dp_loss_v6(t_vals_1, t_vals_0, pdf_1, pdf_0, mus_0, sigmas_0, left_tails_0, part_inside_cells_0, cfg=None):

    """
    this version estimate loss with sum of cdf with only one cdf per section
    """
    # TODO: for speedup - try first estimate the coarse PDF, Than add a correction
    # filter rays if synthetic data is in use, to avoid "no depth" rays
    try:
        if (cfg.dataset.type).lower() == "blender":
            relevent_rows = pdf_1.sum(1) > 1e-10

            # if all rays with no depth..
            if relevent_rows.sum() == 0:
                return relevent_rows.sum()

            pdf_0 = pdf_0[relevent_rows]
            pdf_1 = pdf_1[relevent_rows]
            mus_0 = mus_0[relevent_rows]
            sigmas_0 = sigmas_0[relevent_rows]
            part_inside_cells_0 = part_inside_cells_0[relevent_rows]
            t_vals_1 = t_vals_1[relevent_rows]
            t_vals_0 = t_vals_0[relevent_rows]

    except:
        pass

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
    if cfg.train_params.mse == True:
        m = (estimated_pdf_1 + pdf_1.detach())/2
        loss = torch.nn.functional.kl_div(pdf_1.log().detach(), m) + torch.nn.functional.kl_div(estimated_pdf_1.log(), m)

    else:
        loss = torch.nn.functional.kl_div(estimated_pdf_1.log(), pdf_1.detach())



    if loss.sum() != loss.sum():
        pdb.set_trace()

    return loss  #, estimated_pdf_1
