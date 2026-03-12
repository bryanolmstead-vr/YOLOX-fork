#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# modified to include OBB support, but still compatible with HBB datasets. 
# For OBB datasets, the bbox format is [xc, yc, w, h, theta] where theta is in radians.

"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_obboxes(targets, target_size, M, scale):
    """
    targets: N x 6, [xc, yc, w, h, theta, cls]
    M: 2x3 affine matrix
    """
    num_gts = len(targets)
    twidth, theight = target_size

    new_targets = np.zeros_like(targets)

    for i in range(num_gts):
        xc, yc, w, h, theta, cls = targets[i]

        # compute box corners
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = w / 2
        dy = h / 2
        corners = np.array([
            [dx, dy],
            [-dx, dy],
            [-dx, -dy],
            [dx, -dy]
        ])
        # rotate corners
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_corners = corners @ R.T
        rotated_corners += np.array([xc, yc])

        # apply affine transform
        corners_h = np.hstack([rotated_corners, np.ones((4,1))])  # homogenous
        transformed = (M @ corners_h.T).T  # 4x2

        # fit new rotated rectangle
        rect = cv2.minAreaRect(transformed.astype(np.float32))  # ((xc, yc), (w, h), theta_deg)
        (xc_new, yc_new), (w_new, h_new), theta_deg = rect
        theta_new = np.deg2rad(theta_deg)

        # clip coordinates
        xc_new = np.clip(xc_new, 0, twidth)
        yc_new = np.clip(yc_new, 0, theight)
        w_new = np.clip(w_new, 0, twidth)
        h_new = np.clip(h_new, 0, theight)

        new_targets[i] = [xc_new, yc_new, w_new, h_new, theta_new, cls]

    return new_targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_obboxes(targets, target_size, M, scale)

    return img, targets


def _mirror_obb(image, boxes, thetas, prob=0.5):
    """
    boxes: N x 4 [xc, yc, w, h]
    thetas: N x 1 array of angles in radians
    """
    _, width, _ = image.shape
    if random.random() < prob:
        # Flip image horizontally
        image = image[:, ::-1]

        # Flip box centers
        boxes[:, 0] = width - boxes[:, 0]

        # Flip angles
        thetas[:] = -thetas

    return image, boxes, thetas


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        print("BLO TrainTransform start ===")
        print("Original targets shape:", targets.shape)
        print("Original targets[0]:", targets[0] if len(targets) > 0 else "empty")

        if len(targets) == 0:
            padded_labels = np.zeros((self.max_labels, 6), dtype=np.float32)
            image, _ = preproc(image, input_dim)
            return image, padded_labels
        
        boxes = targets[:, 0:4].copy()
        thetas = targets[:, 4].copy()
        labels = targets[:, 5].copy()

        print("BLO\nBoxes shape:", boxes.shape)
        print("Thetas shape:", thetas.shape)
        print("Labels shape:", labels.shape)

        # HSV
        if random.random() < self.hsv_prob:
            augment_hsv(image)

        # Mirror
        image, boxes, thetas = _mirror_obb(image, boxes, thetas, self.flip_prob)
        print("BLO After mirror: boxes:", boxes.shape, "thetas:", thetas.shape)

        # Resize
        image, r = preproc(image, input_dim)
        boxes *= r
        print("BLO After resize: boxes shape:", boxes.shape)

        # Filter out small boxes
        mask = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask]
        thetas_t = thetas[mask]
        labels_t = labels[mask]

        if len(boxes_t) == 0:
            boxes_t = boxes
            thetas_t = thetas
            labels_t = labels

        print("BLO After filtering: boxes_t:", boxes_t.shape, "thetas_t:", thetas_t.shape, "labels_t:", labels_t.shape)

        targets_t = np.hstack((boxes_t, thetas_t[:, None], labels_t[:, None]))
        padded_labels = np.zeros((self.max_labels, 6), dtype=np.float32)
        padded_labels[:len(targets_t), :] = targets_t[:self.max_labels, :]

        print("BLO\nFinal padded_labels shape:", padded_labels.shape)
        print("Final padded_labels[0]:", padded_labels[0] if len(targets_t) > 0 else "empty")
        print("=== TrainTransform end ===")

        return image, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, targets=None, input_size=(640, 640)):
        img, r = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # Handle labels
        if targets is None or len(targets) == 0:
            padded_labels = np.zeros((1, 6), dtype=np.float32)
        else:
            padded_labels = targets.copy()
            padded_labels[:, :4] *= r  # scale xc, yc, w, h
            # theta and class unchanged

        return img, padded_labels
