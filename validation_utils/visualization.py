import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import io
import os
from PIL import Image
import imageio
import cv2

def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    #np_img = img.detach().cpu().numpy()
    #img = img.clamp(np.percentile(0, np.percentile(np_img, 80))) * 255
    img = img.clamp(0, 1) * 255
    h, w = img.shape
    return img.detach().cpu().numpy().astype(np.uint8).reshape(1, h, w)


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img

def get_hist(tensor, title):

    buff = gen_plot(tensor.squeeze().cpu(), title)

    img = Image.open(buff)

    return np.array(img).transpose(2, 0, 1)

def gen_plot(x, y_list, legend, colors, gt, t_vals, title, tb_mode=False):
    """Create a pyplot plot and save to buffer."""
    if tb_mode:
        w = 7
        h = 5
        dpi = 100
        legend_size = 10
        font_size = 10
    else:
        w = 9
        h = 6
        dpi = 100
        legend_size = 15
        font_size = 15

    plt.figure(figsize=(w, h))

    dy_sctr = float(0.075*y_list[0].max())
    dy_sctr = 0.9*dy_sctr

    for i in range(len(y_list)):
        plt.plot(x, y_list[i], c=colors[i], label=legend[i])

    plt.scatter(x=t_vals[0].detach().cpu(), y=torch.zeros_like(t_vals[0].detach().cpu()), c=colors[0], label="coarse samples")
    plt.scatter(x=t_vals[1].detach().cpu(), y=torch.zeros_like(t_vals[1].detach().cpu())-dy_sctr, c=colors[1], label="fine samples")
    if gt > 0:
        plt.scatter(x=gt, y=dy_sctr, s=100, c="orange", marker='^', label="points of interest")

    plt.legend(fontsize=12, loc="upper left")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title(title, fontsize=font_size)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=400)
    buf.seek(0)
    return buf

def get_density_distribution_plots(output, j, gt_depth, cfg, i=0, tb_mode=False):

    x = torch.linspace(cfg.dataset.near, cfg.dataset.far, 1000)
    gt = gt_depth[j]
    t_vals = [output[0]['t_vals_for_plot'][j], output[1]['t_vals_for_plot'][j]]
    y_list = [output[0]['uniform_incell_pdf_to_plot'][j], output[1]['uniform_incell_pdf_to_plot'][j]]
    color_list = ["b", "m"]
    legend = ['h-c', 'h-f']

    if 'gaussian_incell_pdf_to_plot' in output[1].keys():
        y_list.append(output[1]['gaussian_incell_pdf_to_plot'][j])
        legend.append("f-dd")
        color_list.append("g")
        y_list.append(output[1]['smoothed_gaussian_incell_pdf_to_plot'][j])
        legend.append("smoothed f-dd")
        color_list.append("r")

    title = f"Distributions and samples - ray_{j}"
    title = title + f"- iteration {i}" if tb_mode else title

    buff = gen_plot(x, y_list, legend, color_list, gt, t_vals, title, tb_mode)

    img = Image.open(buff)

    return np.array(img).transpose(2, 0, 1)


def save_validation_images(output_dict, path):

    rgb_coarse = output_dict[0]["rgb"]
    imageio.imwrite(
        os.path.join(path, "rgb_coarse.png"), cast_to_image(rgb_coarse).transpose(1,2,0))

    disp_coarse = output_dict[0]["disp"]
    disp_coarse = cast_to_disparity_image(disp_coarse).squeeze()
    imageio.imwrite(
        os.path.join(path, "coarse.png"), disp_coarse)

    depth = output_dict[0]["depth"]
    depth = cast_to_disparity_image(depth).squeeze()
    imageio.imwrite(
        os.path.join(path, "depth_coarse.png"), depth)

    if 'corrected_disp_map' in output_dict[0].keys():
        disp_coarse_corrected = output_dict[0]["corrected_disp_map"]
        disp_coarse_corrected = cast_to_disparity_image(disp_coarse_corrected).squeeze()
        imageio.imwrite(
            os.path.join(path, "mus.png"), disp_coarse_corrected)

    rgb_fine = output_dict[1]["rgb"]
    imageio.imwrite(
        os.path.join(path, "rgb_fine.png"), cast_to_image(rgb_fine).transpose(1,2,0))

    depth = output_dict[1]["depth"]
    depth = cast_to_disparity_image(depth).squeeze()
    imageio.imwrite(
        os.path.join(path, "depth_fine.png"), depth)

    disp_fine = output_dict[1]["disp"]
    disp_fine = cast_to_disparity_image(disp_fine).squeeze()
    imageio.imwrite(
        os.path.join(path, "fine.png"), disp_fine)

def write_dicts_to_a_file(summary_dict, results_dict, results_file):

    with open(results_file, "w") as f:

        print("average overall results:\n", file=f)

        for key in summary_dict.keys():
            score = sum(summary_dict[key])/len(summary_dict[key])
            print(f"{key}: \t {score:.4}", file = f)

        print("\nper image results:\n", file=f)
        for key1 in results_dict.keys():
            for key2 in results_dict[key1].keys():
                print(f"image {key1} , {key2}: \t {results_dict[key1][key2]:.4}", file=f)