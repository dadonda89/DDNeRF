import torch
from general_utils.nerf_helpers import get_ray_bundle
from data_utils.dataset_helpers import ndc_mipnerf_rays
from data_utils.load_tat import get_rays_single_image



def get_rays_and_target_for_depth_dist(image_target, pose_target, H, W, focal, cfg):
    if cfg.dataset.basedir.split("/")[-1] == "ktm_data":
        ray_origins, ray_directions, ray_rad, depth_s, rgb_target = \
            get_selected_rays_for_ktm_data(image_target, pose_target, H, W, focal, cfg)

    elif cfg.dataset.basedir.split("/")[-1] == "playground_v2":
        ray_origins, ray_directions, ray_rad, depth_s, rgb_target = \
            get_selected_rays_for_playground_v2_data(image_target, pose_target, H, W, focal, cfg)

    elif cfg.dataset.basedir.split("/")[-1] == "fern":
        ray_origins, ray_directions, ray_rad, depth_s, rgb_target = \
            get_selected_rays_for_fern_data(image_target, pose_target, H, W, focal, cfg)

    elif cfg.dataset.basedir.split("/")[-1] == "tat_intermediate_Playground":
        ray_origins, ray_directions, ray_rad, depth_s, rgb_target = \
            get_selected_rays_for_playground_data(image_target, pose_target, H, W, focal, cfg)

    else:
        raise Exception("no depth gt avaliable for this dataset validation")

    return ray_origins, ray_directions, ray_rad, depth_s, rgb_target


def get_selected_rays_for_ktm_data(image_target, pose_target, H, W, focal, cfg):

    ray_origins, ray_directions, radii = get_ray_bundle(H, W, focal, pose_target)

    select_coords = torch.zeros((10, 2))

    # this is a hardcoded indices choosing (manualy choosed from sift features)
    select_coords[0] = torch.tensor([286, 50])  # idx = 0
    select_coords[1] = torch.tensor([126, 297])  # idx = 2
    select_coords[2] = torch.tensor([324, 332])  # idx = 4

    select_coords[3] = torch.tensor([277, 389])  # idx = 20

    select_coords[4] = torch.tensor([558, 447])  # idx = 31
    select_coords[5] = torch.tensor([190, 456])  # idx = 33

    select_coords[6] = torch.tensor([195, 484])  # idx = 40
    select_coords[7] = torch.tensor([250, 543])  # idx = 60
    select_coords[8] = torch.tensor([507, 667])  # idx = 96
    select_coords[9] = torch.tensor([283, 783])  # idx = 99

    select_coords = select_coords.type(torch.int64).to(ray_origins.device)

    # depth values correspondence to the indices choosing
    depth_s = [0.6761, 0.7655, 0.1726, 0.1789, 0.1900, 0.5109, 0.7474, 0.7001, 0.2145, 0.5826]

    ray_origins = ray_origins[select_coords[:, 0], select_coords[:, 1], :]
    ray_directions = ray_directions[select_coords[:, 0], select_coords[:, 1], :]
    ray_rad = radii[select_coords[:, 0], select_coords[:, 1], :].reshape(-1, 1)
    rgb_target = image_target[select_coords[:, 0], select_coords[:, 1]]

    if cfg.dataset.normalize_poses:
        factor =  cfg.dataset.normalize_factor

    else:
        factor = 1

    factor = 18/factor

    depth_s = [x*factor for x in depth_s]

    return ray_origins, ray_directions, ray_rad, depth_s, rgb_target

def get_selected_rays_for_playground_data(image_target, pose_target, H, W, focal, cfg):

    ray_origins, ray_directions, radii = get_rays_single_image(H, W, focal, pose_target)

    select_coords = torch.zeros((11, 2))

    # this is a hardcoded indices choosing (manualy choosed from sift features for images downsampled by 2)
    select_coords[0] = torch.tensor([97, 287])  # idx = 0
    select_coords[1] = torch.tensor([60, 393])  # idx = 17
    select_coords[2] = torch.tensor([31, 465])  # idx = 26

    select_coords[3] = torch.tensor([147, 511])  # idx = 37

    select_coords[4] = torch.tensor([55, 512])  # idx = 39
    select_coords[5] = torch.tensor([343, 527])  # idx = 47

    select_coords[6] = torch.tensor([310, 540])  # idx = 51
    select_coords[7] = torch.tensor([186, 746])  # idx = 76
    select_coords[8] = torch.tensor([254, 882])  # idx = 84
    select_coords[9] = torch.tensor([235, 892])  # idx = 86
    select_coords[10] = torch.tensor([294, 812])  # idx = 83

    select_coords = select_coords.type(torch.int64).to(ray_origins.device)

    # depth values correspondence to the indices choosing
    depth_s = [3.16, 5.27, 5.08, 0.5, 5.12, 0.506, 0.667, 3.32, 0.68, 1.03, 0.68]

    if cfg.dataset.normalize_poses:
        depth_s = [x/cfg.dataset.normalize_factor for x in depth_s]

    ray_origins = ray_origins[select_coords[:, 0], select_coords[:, 1], :]
    ray_directions = ray_directions[select_coords[:, 0], select_coords[:, 1], :]
    ray_rad = radii[select_coords[:, 0], select_coords[:, 1], :].reshape(-1, 1)
    rgb_target = image_target[select_coords[:, 0], select_coords[:, 1]]

    return ray_origins, ray_directions, ray_rad, depth_s, rgb_target

def get_selected_rays_for_playground_v2_data(image_target, pose_target, H, W, focal, cfg):

    ray_origins, ray_directions, radii = get_ray_bundle(H, W, focal, pose_target)

    select_coords = torch.zeros((11, 2))

    # this is a hardcoded indices choosing (manualy choosed from sift features for images downsampled by 2)
    select_coords[0] = torch.tensor([105, 52])  # idx = 0
    select_coords[1] = torch.tensor([16, 55])  # idx = 1
    select_coords[2] = torch.tensor([106, 131])  # idx = 14

    select_coords[3] = torch.tensor([124, 152])  # idx = 17

    select_coords[4] = torch.tensor([85, 180])  # idx = 23
    select_coords[5] = torch.tensor([105, 188])  # idx = 26

    select_coords[6] = torch.tensor([133, 261])  # idx = 55
    select_coords[7] = torch.tensor([184, 377])  # idx = 100
    select_coords[8] = torch.tensor([207, 410])  # idx = 119
    select_coords[9] = torch.tensor([194, 465])  # idx = 122
    select_coords[10] = torch.tensor([35, 395])  # idx = 112

    select_coords = select_coords.type(torch.int64).to(ray_origins.device)

    # depth values correspondence to the indices choosing
    depth_s = [7.91, 9.07, 10.5, 8.37, 12.4, 10.2, 4.65, 3.99, 2.45, 3.44, 11.39]

    ray_origins = ray_origins[select_coords[:, 0], select_coords[:, 1], :]
    ray_directions = ray_directions[select_coords[:, 0], select_coords[:, 1], :]
    ray_rad = radii[select_coords[:, 0], select_coords[:, 1], :].reshape(-1, 1)
    rgb_target = image_target[select_coords[:, 0], select_coords[:, 1]]

    if cfg.dataset.normalize_poses:
        factor = cfg.dataset.normalize_factor

    else:
        factor = 1

    factor = 1/factor

    depth_s = [x*factor for x in depth_s]

    return ray_origins, ray_directions, ray_rad, depth_s, rgb_target



def get_selected_rays_for_fern_data(image_target, pose_target, H, W, focal, cfg):

    ray_origins, ray_directions, radii = get_ray_bundle(H, W, focal, pose_target)
    if cfg.dataset.ndc_rays:
        ray_origins_ndc, ray_directions_ndc, radii_ndc = ndc_mipnerf_rays(H, W, focal, ray_origins, ray_directions)

    select_coords = torch.zeros((13, 2))

    # this is a hardcoded indices choosing (manualy choosed from sift features)
    select_coords[0] = torch.tensor([571, 103])  # idx = 1
    select_coords[1] = torch.tensor([439, 151])  # idx = 2
    select_coords[2] = torch.tensor([52, 432])   # idx = 13
    select_coords[3] = torch.tensor([560, 434])  # idx = 15
    select_coords[4] = torch.tensor([514, 519])   # idx = 17
    select_coords[5] = torch.tensor([529, 629])  # idx = 29
    select_coords[6] = torch.tensor([526, 695])  # idx = 34
    select_coords[7] = torch.tensor([237, 753])  # idx = 39
    select_coords[8] = torch.tensor([208, 870])  # idx = 42
    select_coords[9] = torch.tensor([628, 721])  # idx = 37

    select_coords = select_coords.type(torch.int64).to(ray_origins.device)

    # depth values correspondence to the indices choosing
    depth_s = [2.451, 2.5005, 2.7370, 1.5731, 2.4122, 2.9714, 3.0970, 5.8555, 4.5532, 3.0460]

    #  additional 3 challenging points with no depth values
    select_coords[10] = torch.tensor([550, 721])
    select_coords[11] = torch.tensor([140, 432])
    select_coords[12] = torch.tensor([280, 150])

    depth_s = depth_s + [0, 0, 0]

    ray_origins = ray_origins[select_coords[:, 0], select_coords[:, 1], :]
    ray_directions = ray_directions[select_coords[:, 0], select_coords[:, 1], :]
    ray_rad = radii[select_coords[:, 0], select_coords[:, 1]].reshape(-1, 1)
    rgb_target = image_target[select_coords[:, 0], select_coords[:, 1]]

    if cfg.dataset.ndc_rays:
        depth = torch.tensor(depth_s).to(ray_directions[:, 2].device)
        depth = depth - (1 + ray_origins[:,2])
        depth_s = depth*ray_directions[:,2]/(-1 + depth*ray_directions[:,2])
        depth_s = [float(x.cpu().numpy()) for x in depth_s]

        ray_origins = ray_origins_ndc[select_coords[:, 0], select_coords[:, 1], :]
        ray_directions = ray_directions_ndc[select_coords[:, 0], select_coords[:, 1], :]
        ray_rad = radii_ndc[select_coords[:, 0], select_coords[:, 1]].reshape(-1, 1)


    return ray_origins, ray_directions, ray_rad, depth_s, rgb_target
