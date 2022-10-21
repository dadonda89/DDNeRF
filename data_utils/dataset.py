import torch
import numpy as np
import yaml

from general_utils.nerf_helpers import get_ray_bundle
from data_utils.dataset_helpers import ndc_mipnerf_rays

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
            self.da_origins, self.da_directions, self.da_rad, self.da_depth, self.da_rgb = self.load_depth_analysis_rays(cfg)



        print(f"validation set init finnished, {images.shape[0]} images in the dataset")

    def load_depth_analysis_rays(self, cfg):
        with open(cfg.train_params.depth_analysis_path) as f:
            data = yaml.load(f)

        img_idx = data['img_idx']
        factor = int(data['resized_by'] / cfg.dataset.downsample_factor)

        image_target = self.images[img_idx]
        pose_target = self.poses[img_idx]

        ray_origins, ray_directions, radii = get_ray_bundle(self.H, self.W,
                                                            self.focal, pose_target)

        if cfg.dataset.ndc_rays:
            ray_origins_ndc, ray_directions_ndc, radii_ndc = ndc_mipnerf_rays(self.H, self.W,
                                                          self.focal, ray_origins, ray_directions)

        anotated_data = data['pixels_and_depth']
        select_coords = torch.zeros((len(anotated_data), 2))
        depth_s = []

        for i, c in enumerate(list(anotated_data.values())):
            select_coords[i] = (factor*torch.tensor(c[:2]))
            depth_s.append(c[2])

        select_coords = select_coords.type(torch.int64).to(ray_origins.device)

        ray_origins = ray_origins[select_coords[:, 0], select_coords[:, 1], :]
        ray_directions = ray_directions[select_coords[:, 0], select_coords[:, 1], :]
        ray_rad = radii[select_coords[:, 0], select_coords[:, 1]].reshape(-1, 1)
        rgb_target = image_target[select_coords[:, 0], select_coords[:, 1]]

        if cfg.dataset.ndc_rays:
            depth = torch.tensor(depth_s).to(ray_directions[:, 2].device)
            depth = depth - (1 + ray_origins[:, 2])
            depth_s = depth * ray_directions[:, 2] / (-1 + depth * ray_directions[:, 2])
            depth_s = [float(x.cpu().numpy()) for x in depth_s]

            ray_origins = ray_origins_ndc[select_coords[:, 0], select_coords[:, 1], :]
            ray_directions = ray_directions_ndc[select_coords[:, 0], select_coords[:, 1], :]
            ray_rad = radii_ndc[select_coords[:, 0], select_coords[:, 1]].reshape(-1, 1)

        return ray_origins, ray_directions, ray_rad, depth_s, rgb_target


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

