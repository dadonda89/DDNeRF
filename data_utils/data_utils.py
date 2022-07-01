import torch
import numpy as np


from datasets.load_llff import load_llff_data
from datasets.load_blender import load_blender_data
from datasets.load_tat import load_tat_data
from datasets.dataset import TrainDataset, ValDataset, TrainTatDataset, ValTatDataset


def get_datasets(cfg):

    if cfg.dataset.type.lower() == "blender" or cfg.dataset.type.lower() == "llff":
        train_dataset, val_dataset = load_blender_or_llff_datasets(cfg)

    elif cfg.dataset.type.lower() == "tat":
        train_dataset, val_dataset = load_tat_dataset(cfg)

    else:
        print(f"not familier with dataset type - {cfg.dataset.type.lower()}")
        exit()


    return train_dataset, val_dataset


def load_blender_or_llff_datasets(cfg):

    if cfg.dataset.type.lower() == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )  # hwf = [H, W, F], the camera intrinsic params
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])

        else:
            images = images[..., :3] * images[..., -1:]

        if not cfg.dataset.half_res:
            images = torch.from_numpy(images)

    elif cfg.dataset.type.lower() == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor, bd_factor=cfg.dataset.bd_factor
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]  # the c2w matrix

        if cfg.dataset.reduce_image_number_by != 1:
            idx = np.array([i for i in np.arange(images.shape[0]) if i % cfg.dataset.reduce_image_number_by == 0])
            images = images[idx]
            poses = poses[idx]

        if not isinstance(i_test, list):
            i_test = [i_test]
        if cfg.dataset.llffhold > 0:
            i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(images.shape[0])
                if (i not in i_test and i not in i_val)
            ]
        )
        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)

    _, _, focal = hwf

    if cfg.dataset.normalize_poses:

        for i in range(poses.shape[0]):
            poses[i][:, 3] = poses[i][:, 3] / cfg.dataset.normalize_factor

        cfg.dataset.near = cfg.dataset.near / cfg.dataset.normalize_factor
        cfg.dataset.far = cfg.dataset.far / cfg.dataset.normalize_factor
        cfg.dataset.combined_split = cfg.dataset.combined_split / cfg.dataset.normalize_factor


    train_dataset = TrainDataset(poses[i_train], images[i_train], focal, ndc_rays=cfg.dataset.ndc_rays,
                                 single_image_mode=cfg.dataset.single_image_mode)
    val_dataset = ValDataset(poses[i_val], images[i_val], focal, ndc_rays=cfg.dataset.ndc_rays, cfg=cfg)

    return train_dataset, val_dataset


def load_tat_dataset(cfg):
    train_poses, train_images, train_intrinsics = load_tat_data(cfg.dataset.basedir, "train", cfg)
    val_poses, val_images, val_intrinsics = load_tat_data(cfg.dataset.basedir, "validation", cfg)

    if cfg.dataset.normalize_poses:

        cfg.dataset.near = cfg.dataset.near / cfg.dataset.normalize_factor
        cfg.dataset.far = cfg.dataset.far / cfg.dataset.normalize_factor
        cfg.dataset.combined_split = cfg.dataset.combined_split / cfg.dataset.normalize_factor

    train_dataset = TrainTatDataset(train_poses, train_images, train_intrinsics,
                                    single_image_mode=cfg.dataset.single_image_mode)

    val_dataset = ValTatDataset(val_poses, val_images, val_intrinsics, cfg=cfg)

    return train_dataset, val_dataset


def load_dataset(cfg):

    if cfg.dataset.type.lower() == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )  # hwf = [H, W, F], the camera intrinsic params
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])

        else:
            images = images[..., :3] * images[..., -1:]

        if not cfg.dataset.half_res:
            images = torch.from_numpy(images)

    elif cfg.dataset.type.lower() == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor, bd_factor=cfg.dataset.bd_factor
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]  # the c2w matrix

        if cfg.dataset.reduce_image_number_by != 1:
            idx = np.array([i for i in np.arange(images.shape[0]) if i % cfg.dataset.reduce_image_number_by == 0])
            images = images[idx]
            poses = poses[idx]

        if not isinstance(i_test, list):
            i_test = [i_test]
        if cfg.dataset.llffhold > 0:
            i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(images.shape[0])
                if (i not in i_test and i not in i_val)
            ]
        )
        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)

    elif cfg.dataset.type.lower() == "tat":
        train_poses, train_images, train_H, train_W, train_focal = load_tat_data(cfg.dataset.basedir, "train")
        val_poses, val_images, val_H, val_W, val_focal = load_tat_data(cfg.dataset.basedir, "validation")

        poses = torch.cat((train_poses, val_poses))
        images = torch.cat((train_images, val_images))

        if (train_H != val_H) or (train_W != val_W) or (train_focal != val_focal) :
            print("val and train not have the same data preperties")
            exit()

        hwf = (train_H, train_W, train_focal)

        i_train = np.arange(train_images.shape[0])
        i_val = np.arange(val_images.shape[0]) + train_images.shape[0]

    else:
        print(f"not familier with dataset type - {cfg.dataset.type.lower()}")

    H, W, focal = hwf
    H, W = int(H), int(W)

    if cfg.dataset.normalize_poses:
        if not cfg.dataset.normalize_factor:
            cfg.dataset.normalize_factor = 0
            for i in range(poses.shape[0]):

                dist = torch.sqrt(poses[i][:, 3].T @ poses[i][:, 3])

                if dist > cfg.dataset.normalize_factor:
                    cfg.dataset.normalize_factor = float(dist)

        for i in range(poses.shape[0]):
            poses[i][:, 3] = poses[i][:, 3] / cfg.dataset.normalize_factor


    return poses, images, H, W, focal, i_train, i_val


