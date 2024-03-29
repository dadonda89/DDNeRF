import math
import numpy as np
from typing import Optional

import torch

def cast_rays(t_vals, origins, directions, radii, ray_shape='cone', diag=True):
  """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

  Args:
    t_vals: float array, the "fencepost" distances along the ray.
    origins: float array, the ray origin coordinates.
    directions: float array, the ray direction vectors.
    radii: float array, the radii (base radii for cones) of the rays.
    ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
    diag: boolean, whether or not the covariance matrices should be diagonal.

  Returns:
    a tuple of arrays of means and covariances.
  """
  t0 = t_vals[..., :-1]
  t1 = t_vals[..., 1:]
  if ray_shape == 'cone':
    gaussian_fn = conical_frustum_to_gaussian
  elif ray_shape == 'cylinder':
    gaussian_fn = cylinder_to_gaussian
  else:
    assert False
  means, covs = gaussian_fn(directions, t0, t1, radii, diag)
  means = means + origins[..., None, :]
  return means, covs


def lift_gaussian(d, t_mean, t_var, r_var, diag):
  """Lift a Gaussian defined along a ray to 3D coordinates."""
  mean = d[..., None, :] * t_mean[..., None]

  d_mag_sq = torch.maximum(torch.tensor(1e-10), torch.sum(d**2, axis=-1, keepdims=True))

  if diag:
    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
  else:
    d_outer = d[..., :, None] * d[..., None, :]
    eye = torch.eye(d.shape[-1])
    null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
    t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
    xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
    cov = t_cov + xy_cov
    return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
  """Approximate a conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.

  Args:
    d: jnp.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

  Returns:
    a Gaussian (mean and covariance).
  """
  if stable:
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                      (3 * mu**2 + hw**2)**2)
    r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                              (hw**4) / (3 * mu**2 + hw**2))
  else:
    t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
    r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
    t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_var = t_mosq - t_mean**2
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
  """Approximate a cylinder as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and radius is the
  radius. Does not renormalize `d`.

  Args:
    d: jnp.float32 3-vector, the axis of the cylinder
    t0: float, the starting distance of the cylinder.
    t1: float, the ending distance of the cylinder.
    radius: float, the radius of the cylinder
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
  t_mean = (t0 + t1) / 2
  r_var = radius**2 / 4
  t_var = (t1 - t0)**2 / 12
  return lift_gaussian(d, t_mean, t_var, r_var, diag)

def integrated_pos_enc(x_coord,  max_deg=16, min_deg=0, diag=True):
  """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

  Args:
    x_coord: a tuple containing: x, jnp.ndarray, variables to be encoded. Should
      be in [-pi, pi]. x_cov, jnp.ndarray, covariance matrices for `x`.
    min_deg: int, the min degree of the encoding.
    max_deg: int, the max degree of the encoding.
    diag: bool, if true, expects input covariances to be diagonal (full
      otherwise).

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if diag:
    x, x_cov_diag = x_coord
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).to(x.device)
    shape = list(x.shape[:-1]) + [-1]
    y = torch.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)
  else:
    x, x_cov = x_coord
    num_dims = x.shape[-1]
    basis = torch.concatenate(
        [2**i * torch.eye(num_dims) for i in range(min_deg, max_deg)], 1)
    y = torch.matmul(x, basis)
    # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
    # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
    y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)

  return expected_sin(
      torch.cat([y, y + 0.5 * torch.tensor(np.pi)], dim=-1),
      torch.cat([y_var] * 2, dim=-1))[0]

def expected_sin(x, x_var):
  """Estimates mean and variance of sin(z), z ~ N(x, var)."""
  # When the variance is wide, shrink sin towards zero.
  y = torch.exp(-0.5 * x_var) * safe_sin(x)
  y_var = torch.maximum(
      torch.tensor(0), 0.5 * (1 - torch.exp(-2 * x_var) * safe_cos(2 * x)) - y**2)
  return y, y_var


def safe_trig_helper(x, fn, t=100 * torch.tensor(np.pi)):
  return fn(torch.where(torch.abs(x) < t, x, x % t))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.sin)


def bins_for_percentage(weights, percentage):
  """
  return the number of bins required for "percentage" part of the information
  """
  pdf = weights/(weights.sum(dim=1).reshape(-1,1))

  info_sorted = torch.sort(pdf, descending=True)[0]

  info_sum = torch.cumsum(info_sorted[..., :-1], dim=-1)

  bfp = (info_sum < percentage).sum(1)+1

  return bfp

def approximate_cdf_old(x):
  """
  approximation to the cdf calculation for normal distribution
  """
  pi = torch.tensor(np.pi)

  ans = 0.5 * torch.tanh((39 * x / (2 * torch.sqrt(2 * pi))) - (111 / 2) * torch.arctan(35 * x / (111 * torch.sqrt(2 * pi)))) + 0.5

  return ans

def approximate_cdf(x):
  """
  approximation to the cdf calculation for normal distribution
  """
  sqrt2 = torch.sqrt(torch.tensor(2.0, device=x.device))

  return 0.5 * (1+torch.erf(x/sqrt2))


def approximate_inverse_cdf(x):
  """
  approximation to the inverse cdf calculation for normal distribution
  """
  sqrt2 = torch.sqrt(torch.tensor(2, device=x.device))

  return sqrt2 * torch.erfinv(2*x - 1)

def get_uniform_incell_pdf(t_vals, weights, cfg):

    """
    estimated the pdf along a ray with uniform in cell distribution
    """

    pdf = (weights/torch.sum(weights, dim=-1, keepdim=True)).cpu()

    bins = torch.linspace(cfg.dataset.near, cfg.dataset.far, 1000).reshape(1,-1)

    estmated_pdf = torch.zeros(pdf.shape[0], bins.shape[1])

    for i in range(pdf.shape[1]):

      start = t_vals[:, i].reshape(-1, 1).cpu()
      end = t_vals[:, i + 1].reshape(-1, 1).cpu()

      relevant_cells = (bins >= start) * (bins < end)

      divided_by = relevant_cells.sum(1).reshape(-1,1) # to get estimated pdf per section

      estmated_pdf += relevant_cells*pdf[:,i].reshape(-1,1)/divided_by

    return estmated_pdf


def get_gaussian_incell_pdf(t_vals, weights, mus, sigmas, part_inside_cells, cfg):
  """
  estimated the pdf along a ray with gaussian in cell distribution
  """

  pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

  # transform mu, sigma from section space to ray space
  mus = (t_vals[:, :-1] + mus * (t_vals[:, 1:] - t_vals[:, :-1])).cpu()
  sigmas = (sigmas * (t_vals[:, 1:] - t_vals[:, :-1])).cpu()

  partitions = torch.linspace(cfg.dataset.near, cfg.dataset.far, 1001).reshape(1, -1)

  estimated_pdf = torch.zeros(pdf.shape[0], partitions.shape[1]-1)

  for i in range(pdf.shape[1]):

    start = t_vals[:, i].reshape(-1, 1).cpu()

    end = t_vals[:, i + 1].reshape(-1, 1).cpu()

    relevant_cells = (partitions[:, :-1] >= start) * (partitions[:, 1:] <= end)

    x0, x1 = partitions[:, :-1].reshape(1, -1), partitions[:, 1:].reshape(1, -1)

    z0 = (x0 - mus[:, i].reshape(-1, 1))/sigmas[:, i].reshape(-1, 1)
    z1 = (x1 - mus[:, i].reshape(-1, 1))/sigmas[:, i].reshape(-1, 1)

    # estimate pdf for each tiny cell
    cdf0 = approximate_cdf(z0)
    cdf1 = approximate_cdf(z1)

    cells_cdf = (cdf1 - cdf0)*(1/part_inside_cells[:, i].cpu().reshape(-1, 1))

    relevant_cells_cdf = relevant_cells*cells_cdf*pdf[:, i].cpu().reshape(-1,1)

    estimated_pdf += relevant_cells_cdf

  zeros_idx = torch.where(estimated_pdf == 0)

  estimated_pdf[zeros_idx[0], zeros_idx[1]] = (estimated_pdf[zeros_idx[0], torch.min(zeros_idx[1] + 1, torch.ones_like(zeros_idx[1])*estimated_pdf.shape[1] - 1)] + estimated_pdf[zeros_idx[0], torch.max(zeros_idx[1] - 1, torch.zeros_like(zeros_idx[1]))])/2

  return estimated_pdf
