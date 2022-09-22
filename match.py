"""Main program"""

import os
import sys
from math import ceil
from tqdm import tqdm
import numpy as np
import openslide
from utils import matcher, gen_patch_info
from utils import crop_img_to_arr, get_crop_kpt_des, integrate_coords
from utils import crop_finetune_img, fine_tune, plot_comp_img


def match(dir_wsi: str, dir_patch: str,
          level=0, ori_crop_size=(3432, 1884),
          dsample_match=8, threshold=0.4,
          finetune_range=60, finetune_thresh=0.9,
          debug_mode=False):
    """
    Args:
        dir_wsi: string,
            directory of whole slide images.
        dir_patch: string,
            directory of patch images.
        level: integer,
            level number for cropping patches from WSI.
        ori_crop_size: tuple of two integers,
            original size of cropped patches.
        dsample_match: integer,
            downsampling rate for matching.
        threshold: float (0~1),
            threshold for SIFT algorithm.
        finetune_range: integer,
            the pixel range for validating matched patch.
        finetune_thresh: float (0~1),
            threshold for fine-tuning.
        debug_mode: boolean,
            whether to show cropping and patch image.

    Return:
        A dictionary contains matching information.
    """
    ori_crop_width, ori_crop_height = ori_crop_size
    name_list_patch = os.listdir(dir_patch)
    name_list_patch.sort()
    patch_info_dict = gen_patch_info(
        dir_patch, name_list_patch, ori_crop_size, dsample_match)
    name_list_wsi = os.listdir(dir_wsi)
    name_list_wsi.sort()

    matched_dict = {}
    for name_wsi in name_list_wsi:
        slide = openslide.OpenSlide(os.path.join(dir_wsi, name_wsi))

        ratio = int(slide.level_dimensions[0][0]/slide.level_dimensions[level][0])
        crop_width = int(ori_crop_width*ratio)
        crop_height = int(ori_crop_height*ratio)

        print("Matching patches in slide:", name_wsi)

        if debug_mode:
            (slide.read_region(
                (0, 0),
                len(slide.level_dimensions) - 1,
                slide.level_dimensions[-1])).show()

        wsi_width, wsi_height = slide.level_dimensions[0]

        xy_dict = {}

        total = ceil(wsi_height/crop_height)*ceil(wsi_width/crop_width)
        with tqdm(total=total, file=sys.stdout) as pbar:
            for height_i in range(0, wsi_height, crop_height):
                for width_i in range(0, wsi_width, crop_width):
                    coord = (width_i, height_i)
                    kpt_crop, des_crop = get_crop_kpt_des(
                        slide, coord, ori_crop_size, level, dsample_match)

                    for name_patch, patch_info in patch_info_dict.items():
                        kpt_patch = patch_info["kpt"]
                        des_patch = patch_info["des"]
                        matches = matcher(
                            kpt_crop, des_crop,
                            kpt_patch, des_patch, threshold)
                        matches = matches*ratio*dsample_match

                        if len(matches) > 0:
                            if name_patch not in xy_dict:
                                xy_dict[name_patch] = []
                            xy_dict[name_patch].append(
                                (matches[:, :2]
                                + np.array([width_i, height_i])
                                - matches[:, 2:4]))
                    pbar.update(1)

        for name_patch, xy_val in xy_dict.items():
            print("Checking and fine-tuning patch:",
                name_patch, "...")
            coord = integrate_coords(xy_val)

            crop_img = crop_finetune_img(
                slide, coord, ori_crop_size, ratio,
                level, dsample_match, finetune_range)

            patch_img = patch_info_dict[name_patch]["img"]

            d_x, d_y, best_crop_img = fine_tune(
                crop_img, patch_img, ori_crop_size, dsample_match,
                debug_mode, finetune_range, return_best_crop_img=True)

            dist_norm = 1 - np.sqrt(d_x**2 + d_y**2)/finetune_range
            if dist_norm > finetune_thresh:
                width_i, height_i = coord
                new_width_i = width_i + d_x*ratio*dsample_match
                new_height_i = height_i + d_y*ratio*dsample_match
                print("Matched! Patch:", name_patch,
                    "is in slide:", name_wsi, ", start with:",
                    new_width_i, new_height_i)
                info_dict = {}
                info_dict["WSI name"] = name_wsi
                info_dict["start coord"] = [new_width_i, new_height_i]
                matched_dict[name_patch] = info_dict

                if debug_mode:
                    plot_comp_img(best_crop_img, patch_img)

                del patch_info_dict[name_patch]
            else:
                print("Not matched!")
                if debug_mode:
                    crop_img = crop_img_to_arr(
                        slide, coord, ori_crop_size, level, dsample_match)
                    plot_comp_img(crop_img, patch_img)

        slide.close()
        if len(patch_info_dict) <= 0:
            break
    return matched_dict


def save_json(matched_dict: dict, path: str):
    """
    Args:
        matched_dict: dictionary,
            the return from match().
        path: string,
            path of json file.
    """
    with open(path, "w", encoding="utf-8") as file:
        file.write(
            str(matched_dict).replace("'", "\""))
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir_wsi", help="directory of whole slide images")
    parser.add_argument("dir_patch", help="directory of patch images")
    parser.add_argument("-lv", "--level",
                        type=int, default=0,
                        help="level number for cropping patches from WSI")
    parser.add_argument("-ow", "--original_width",
                        type=int, default=3432,
                        help="original width of cropped patches")
    parser.add_argument("-oh", "--original_height",
                        type=int, default=1884,
                        help="original height of cropped patches")
    parser.add_argument("-ds", "--dsample",
                        type=int, default=8,
                        help="downsampling rate for matching")
    parser.add_argument("-t", "--threshold",
                        type=float, default=0.4,
                        help="threshold for SIFT algorithm (0~1)")
    parser.add_argument("-fr", "--finetune_range",
                        type=int, default=60,
                        help="the pixel range for validating matched patch")
    parser.add_argument("-ft", "--finetune_thresh",
                        type=int, default=0.9,
                        help="threshold for fine-tuning (0~1)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="display cropping and patch image")
    parser.add_argument("-o", "--output",
                        default="output.json",
                        help="path of output file")
    args = parser.parse_args()

    dir_wsi = args.dir_wsi
    dir_patch = args.dir_patch
    level = args.level
    ori_crop_size = (args.original_width, args.original_height)
    dsample_match = args.dsample
    threshold = args.threshold
    finetune_range = args.finetune_range
    finetune_thresh = args.finetune_thresh
    debug_mode = args.verbose
    path_output = args.output

    matched_dict = match(
        dir_wsi, dir_patch,
        level, ori_crop_size,
        dsample_match, threshold,
        finetune_range, finetune_thresh,
        debug_mode)

    if save_json(matched_dict, path_output):
        print(f"Saved output file to \"{path_output}\" successfully")
