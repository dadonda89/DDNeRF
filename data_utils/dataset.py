import torch
import numpy as np

from general_utils.nerf_helpers import get_ray_bundle
from data_utils.load_tat import get_rays_single_image
from data_utils.dataset_helpers import ndc_mipnerf_rays
from data_utils.validation import get_rays_and_target_for_depth_dist

class TrainDataset():
    """
    this object generate rays for training, it is different from the regular function because its take rays from multiple images in each iteration
    """
    def __init__(self, poses, images, focal, ndc_rays=False, single_image_mode=False):

        self.images = images
        self.poses = poses

        self.origins = []
        self.directions = []
        self.radii = []
        self.target = []
        self.ndc = ndc_rays
        self.H = images.shape[1]
        self.W = images.shape[2]
        self.focal = focal
        self.near = 1
        self.single_image_mode = single_image_mode

        for i in range(len(images)):
            ray_origins, ray_directions, radii = get_ray_bundle(self.H, self.W, focal, self.poses[i])
            if self.ndc:
                ray_origins, ray_directions, radii = ndc_mipnerf_rays(self.H, self.W, self.focal,
                                                                      ray_origins,
                                                                      ray_directions, self.near)

            self.origins.append(ray_origins.reshape(-1, 3))
            self.directions.append(ray_directions.reshape(-1, 3))
            self.radii.append(radii.reshape(-1, 1))
            self.target.append(self.images[i].reshape(-1,3))

        if not self.single_image_mode:
            self.origins = torch.vstack(self.origins)
            self.directions = torch.vstack(self.directions)
            self.radii = torch.vstack(self.radii)
            self.target = torch.vstack(self.target)

        number_of_rays = len(self.origins)*self.H*self.W if self.single_image_mode else self.origins.shape[0]

        print(f"training set init finnished, {number_of_rays} rays in the dataset")

    def get_training_rays_for_next_iter(self, number_of_rays, device):
        if not self.single_image_mode:
            idxs = np.random.choice(len(self.origins), number_of_rays)
            return self.origins[idxs].to(device), self.directions[idxs].to(device), self.radii[idxs].to(device), self.target[idxs].to(device)

        else:
            img_idx = int(np.random.choice(len(self.origins), 1))
            idxs = np.random.choice(len(self.origins[img_idx]), number_of_rays)
            return self.origins[img_idx][idxs].to(device), self.directions[img_idx][idxs].to(device), self.radii[img_idx][idxs].to(device), \
                   self.target[img_idx][idxs].to(device)



class ValDataset():
    """
    this object generate rays for training, it is different from the regular function because its take rays from multiple images in each iteration
    """
    def __init__(self, poses, images, focal, ndc_rays=False, cfg=None):

        self.images = images
        self.poses = poses

        self.origins = []
        self.directions = []
        self.radii = []
        self.target = []
        self.ndc = ndc_rays
        self.H = images.shape[1]
        self.W = images.shape[2]
        self.focal = focal
        self.near = 1
        self.current_idx = 0

        if cfg.train_params.depth_analysis_rays:
            self.da_origins, self.da_directions, self.da_rad, self.da_depth, self.da_rgb = \
                get_rays_and_target_for_depth_dist(self.images[0], self.poses[0], self.H, self.W, self.focal, cfg)



        print(f"validation set init finnished, {images.shape[0]} images in the dataset")


    def get_next_validation_rays(self, device):

        ray_origins, ray_directions, radii = get_ray_bundle(self.H, self.W, self.focal, self.poses[self.current_idx])
        if self.ndc:
            ray_origins, ray_directions, radii = ndc_mipnerf_rays(self.H, self.W, self.focal,
                                                                  ray_origins,
                                                                  ray_directions, self.near)
        gt_image = self.images[self.current_idx]

        self.current_idx = (self.current_idx+1)%(self.images.shape[0])

        return ray_origins.to(device), ray_directions.to(device), radii.to(device), gt_image.to(device)

    def get_current_regular_validation_rays(self, device):

        ray_origins, ray_directions, radii = get_ray_bundle(self.H, self.W, self.focal, self.poses[self.current_idx])

        return ray_origins.to(device), ray_directions.to(device), radii.to(device)

    def get_depth_analysis_rays(self, device):
        return self.da_origins.to(device), self.da_directions.to(device), self.da_rad.to(device), \
               self.da_depth, self.da_rgb.to(device)

class TrainTatDataset(TrainDataset):
    """
    this object generate rays for training, it is different from the regular function because its take rays from multiple images in each iteration
    """
    def __init__(self, poses, images, intrinsics, single_image_mode=True):

        self.images = images
        self.poses = poses
        self.intrinsics = intrinsics

        self.origins = []
        self.directions = []
        self.radii = []
        self.target = []

        self.H = images.shape[1]
        self.W = images.shape[2]

        self.single_image_mode = single_image_mode

        for i in range(len(images)):
            ray_origins, ray_directions, radii = get_rays_single_image(self.H, self.W, self.intrinsics[i], self.poses[i])

            self.origins.append(ray_origins.reshape(-1, 3))
            self.directions.append(ray_directions.reshape(-1, 3))
            self.radii.append(radii.reshape(-1, 1))
            self.target.append(self.images[i].reshape(-1,3))

        if not self.single_image_mode:
            self.origins = torch.vstack(self.origins)
            self.directions = torch.vstack(self.directions)
            self.radii = torch.vstack(self.radii)
            self.target = torch.vstack(self.target)

        number_of_rays = len(self.origins)*self.H*self.W if self.single_image_mode else self.origins.shape[0]

        print(f"training set init finnished, {number_of_rays} rays in the set")



class ValTatDataset(ValDataset):
    """
    this object generate rays for training, it is different from the regular function because its take rays from multiple images in each iteration
    """
    def __init__(self, poses, images, intrinsics, cfg=None):

        self.images = images
        self.poses = poses
        self.intrinsics = intrinsics

        self.H = images.shape[1]
        self.W = images.shape[2]

        self.current_idx = 0

        if cfg.train_params.depth_analysis_rays:
            self.da_origins, self.da_directions, self.da_rad, self.da_depth, self.da_rgb = \
                get_rays_and_target_for_depth_dist(self.images[0], self.poses[0], self.H, self.W, self.intrinsics[0], cfg)


        print(f"validation set init finnished, {images.shape[0]} images in the set")

    def get_next_validation_rays(self, device):

        ray_origins, ray_directions, radii = get_rays_single_image(self.H, self.W, self.intrinsics[self.current_idx],
                                                            self.poses[self.current_idx])

        gt_image = self.images[self.current_idx]

        self.current_idx = (self.current_idx + 1) % (self.images.shape[0])

        return ray_origins.to(device), ray_directions.to(device), radii.to(device), gt_image.to(device)

