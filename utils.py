"""Utilities for patchmatching algorithm."""

import sys
import os
from packaging import version
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_image(path: str, size=None):
    """
    Args:
        path: path of an image.
        size: target size.

    Return:
        np.array.
    """
    img = Image.open(path)
    img = img.convert("L")
    if size is not None:
        img = img.resize(size)
    img = np.array(img)
    return img


def sift(img):
    """Scale-invariant feature transform algorithm.

    Args:
        img: np.array image.

    Returns:
        (kpt, des)
    """
    if version.parse(cv2.__version__) > version.parse("3.4.2.16"):
        sift_detector= cv2.SIFT_create()
    else:
        sift_detector= cv2.xfeatures2d.SIFT_create()
    kpt, des = sift_detector.detectAndCompute(img, None)
    return kpt, des


def matcher(kpt1, des1, kpt2, des2, threshold=0.4):
    """
    Args:
        kpt1: keypoints of image A.
        des1: vectors of image A.
        kpt2: keypoints of image B.
        des2: vectors of image B.
        threshold: threshold for matching kpt and des.
    """
    # BFMatcher with default params
    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for match_m, match_n in matches:
        if match_m.distance < threshold*match_n.distance:
            good.append([match_m])

    matches = []
    for pair in good:
        matches.append(
            list(kpt1[pair[0].queryIdx].pt + kpt2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches


def gen_patch_info(dir_patch: str, name_list_patch: list,
                   size: tuple, dsample=4):
    """
    Args:
        dir_patch: directory of all patches.
        name_list_patch: file names of all patches.
        size: size for cropping patches.
        dsample: downsampling rate for matching.

    Returns:
        A dictionary.
    """
    ori_crop_width, ori_crop_height = size

    patch_info_dict = {}
    for name in name_list_patch:
        info_dict = {}
        patch_img = read_image(
            os.path.join(dir_patch, name),
            size=(ori_crop_width//dsample,
                  ori_crop_height//dsample))
        info_dict["img"] = patch_img
        kpt, des = sift(patch_img)
        info_dict["kpt"] = kpt
        info_dict["des"] = des
        patch_info_dict[name] = info_dict

    return patch_info_dict


def crop_img_to_arr(slide, coord: tuple, size: tuple,
                    level=1, dsample=4):
    """
    Args:
        slide: a return from openslide.OpenSlide().
        coord: coordinate for cropping patch.
        size: size for cropping patch.
        level: level number for cropping patch from WSI.
        dsample: downsampling rate for matching.

    Returns:
        np.array.
    """
    width_i, height_i = coord
    ori_crop_width, ori_crop_height = size

    # prevent glitch
    for d_x in range(10):
        width_i += d_x
        crop_img = slide.read_region(
            (width_i, height_i), level, (ori_crop_width, ori_crop_height))
        crop_img = crop_img.resize(
            (ori_crop_width//dsample, ori_crop_height//dsample))
        crop_img = np.array(crop_img.convert("L"))
        if crop_img.mean() > 50:
            break

    return crop_img


def get_crop_kpt_des(slide, coord, size, level=1, dsample=4):
    """
    Args:
        slide: a return from openslide.OpenSlide().
        coord: coordinate for cropping patch.
        size: size for cropping patch.
        level: level number for cropping patch from WSI.
        dsample: downsampling rate for matching.

    Returns:
        (kpt, des)
    """
    crop_img = crop_img_to_arr(slide, coord, size, level, dsample)
    kpt_crop, des_crop = sift(crop_img)

    return kpt_crop, des_crop


def integrate_coords(xy_lists):
    """
    Args:
        xy_lists: xy coordinates.
    """
    xy_list = np.vstack(xy_lists)

    dist = np.sqrt(xy_list[:, 0]**2 + xy_list[:, 1]**2)
    residue = abs(dist - dist.mean())
    mask_inlier = residue <= dist.std()*1.5

    xy_list = xy_list[mask_inlier]

    x_val, y_val = np.mean(xy_list, axis=0)
    x_val, y_val = int(round(x_val)), int(round(y_val))

    return x_val, y_val


def crop_finetune_img(slide, coord: tuple, size:tuple, ratio,
                      level=1, dsample=4, finetune_range=60):
    """
    Args:
        slide: a return from openslide.OpenSlide().
        coord: coordinate for cropping patch.
        size: size for cropping patch.
        ratio: ratio of level 0 to a specific level.
        level: level number for cropping patch from WSI.
        dsample: downsampling rate for matching.
        finetune_range: pixel range for validating matched patch.

    Returns:
        (kpt, des)
    """
    half_finetune_range = finetune_range//2
    width_i, height_i = coord
    ori_crop_width, ori_crop_height = size

    coord_finetune = (width_i - half_finetune_range*ratio*dsample,
                      height_i - half_finetune_range*ratio*dsample)
    size_finetune = (ori_crop_width + finetune_range*dsample,
                     ori_crop_height + finetune_range*dsample)

    crop_img = crop_img_to_arr(
        slide, coord_finetune, size_finetune, level, dsample)
    return crop_img


def fine_tune(crop_img, patch_img, size,
              dsample=4, debug_mode=True,
              finetune_range=60, return_best_crop_img=True):
    """
    Args:
        crop_img: image cropped from WSI.
        patch_img: patch image.
        size: size for cropping patch.
        dsample: downsampling rate for matching.
        debug_mode: whether to show cropping and patch image.
        finetune_range: pixel range for validating matched patch.

    Returns:
        (d_x, d_y, [best cropping image])
    """
    half_finetune_range = finetune_range//2
    ori_crop_width, ori_crop_height = size

    min_value = None
    goodx = 0
    goody = 0
    best_crop_img = None

    total = finetune_range*finetune_range
    with tqdm(total=total, file=sys.stdout) as pbar:
        for i_h in range(finetune_range):
            for i_w in range(finetune_range):
                patch_crop = crop_img[i_h:i_h + ori_crop_height//dsample,
                                      i_w:i_w + ori_crop_width//dsample]
                patch_crop = np.maximum(patch_crop, 1e-7)
                entropy = (- patch_img*np.log(patch_crop)).mean()

                if min_value is None or entropy < min_value:
                    min_value = entropy
                    goodx, goody = i_w, i_h
                    best_crop_img = patch_crop
                pbar.update(1)

    d_x = goodx - half_finetune_range
    d_y = goody - half_finetune_range

    if debug_mode:
        print("min cross entropy:", min_value)
        print("dist:", np.sqrt(d_x**2 + d_y**2))

    if return_best_crop_img:
        return d_x, d_y, best_crop_img
    return d_x, d_y


def plot_comp_img(crop_img, patch_img, figsize=(16, 4)):
    """
    Args
        crop_img: image cropped from WSI.
        patch_img: patch image.
        figsize: figsize for pyplot.
    """
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(crop_img, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.imshow(patch_img, cmap="gray")
    plt.show()
