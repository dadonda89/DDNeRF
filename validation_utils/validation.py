import torch
import numpy as np
import cv2
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim

def calc_ssim(image, target):

    image = image.cpu().numpy().astype(np.float32)
    target = target.cpu().numpy().astype(np.float32)

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    (ssim_v1, diff) = compare_ssim(target_gray, image_gray, full=True)
    ssim_v2 = ssim(target_gray, image_gray, data_range=image_gray.max() - image_gray.min())

    return ssim_v1, ssim_v2