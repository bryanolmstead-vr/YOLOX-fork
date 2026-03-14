#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp
# Import OBB dataset and transforms
from yolox.data.datasets.obb import OBBDataset
from yolox.data import TrainTransform, ValTransform

# yolox_s_obb_3classes.py
# input data has 3 classes: candy, cards, cheeto
# format: [id, x_center, y_center, width, height, angle]
# x_center, y_center, width, height are pixels [0,640]
# angle is in degrees, range [-180, 180]

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path (assumes we are in YOLOX directory)
        self.data_dir = "../YOLOX-OneShot/datasets/OBB360"
        # BLO - modify these files to include aabb w,h but store in obb format
        self.train_ann = "instances_obb2aabb_train2017.json"
        self.val_ann = "instances_obb2aabb_val2017.json"

        self.num_classes = 3
        self.cls_names = ["candy", "cards", "cheeto"]

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
        self.weight_decay = 5e-4
        self.warmup_epochs = 3

        # Input image size & multi-scale
        self.input_size = (640, 640)    # base input image size
        self.random_size = (18, 22)     # scale ranges 18x32 to 22x32: 576 to 704 vs 640
        
        # AUGMENTATIONS

        # Probabilities
        self.mosaic_prob = 0.7
        self.mixup_prob = 0.5
        self.hsv_prob = 1.0
        self.flip_prob = 0.0   # disable horizontal flip

        # RandomAffine parameters
        self.degrees = 180.0   # full rotation
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0

        # TURN OFF ALL AUGMENTATIONS  
        print("BLO - Turned off all augmentations")
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0 
        self.degrees = 0.0
        self.translate = 0.0
        self.scale = (1.0, 1.0)
        self.shear = 0.0
        self.perspective = 0.0

        # Disable mosaic in final epochs
        self.no_aug_epochs = 5

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        print("BLO - loading training dataset")
        return OBBDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train2017",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type
        )
    
    def get_eval_dataset(self, **kwargs):
        print("BLO - loading validation dataset")
        legacy = kwargs.get("legacy", False)
        return OBBDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
