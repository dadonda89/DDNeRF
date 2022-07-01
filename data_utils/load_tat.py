import os
import torch
import numpy as np
import glob
import imageio

def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    directions = torch.from_numpy(rays_d.reshape(H, W, 3))
    dx = torch.sqrt(
        torch.sum((directions[:-1, :, :] - directions[1:, :, :]) ** 2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)

    return torch.from_numpy(rays_o.reshape(H, W, -1)), torch.from_numpy(rays_d.reshape(H, W, -1)), radii.reshape(H, W, -1)


def load_tat_data(basedir, split, cfg, skip=1, only_img_files=False):
    """
    split: str, train/validation/test
    """

    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    split_dir = os.path.join(basedir, split)

    if only_img_files:
        img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
        return img_files

    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])

    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    cam_cnt = len(pose_files)

    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        img_files = img_files[::skip]
        assert(len(img_files) == cam_cnt)
    else:
        img_files = [None, ] * cam_cnt
    # assume all images have the same size as training image
    train_imgfile = find_files('{}/train/rgb'.format(basedir), exts=['*.png', '*.jpg'])[0]
    train_im = imageio.imread(train_imgfile)
    H, W = train_im.shape[:2]

    poses = np.zeros((cam_cnt, 4, 4))
    intrinsics = np.zeros((cam_cnt, 4, 4))
    images = np.zeros((cam_cnt, H, W, 3))
    # create ray samplers
    ray_samplers = []
    for i in range(cam_cnt):
        if i == 0:
            focal = parse_txt(intrinsics_files[i])[0, 0]
        else:
            f = parse_txt(intrinsics_files[i])[0, 0]
            if f != focal:
                print("focal lenght issue")
                exit()

        intrinsics[i] = parse_txt(intrinsics_files[i])
        poses[i] = parse_txt(pose_files[i])
        images[i] = imageio.imread(img_files[i]).astype(np.float64) / 255
        H_i, W_i = images[i].shape[:2]
        if (H != H_i) or (W != W_i):
            print("image size issue")
            exit()

    if cfg.dataset.normalize_poses:
        for i in range(poses.shape[0]):
            poses[i][:, 3] = poses[i][:, 3] / cfg.dataset.normalize_factor

    print(f"{cam_cnt} images was loaded as {split} data")
    print(f"H={H}, W={W}, focal={focal}")
    return torch.from_numpy(poses.astype(np.float64)), torch.from_numpy(images), torch.from_numpy(intrinsics.astype(np.float64))
