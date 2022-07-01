import torch
import numpy as np
from nerf_utils.nerf_helpers_and_samplers import get_minibatches

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
