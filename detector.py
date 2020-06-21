#!/bin/env python
# -*- coding: utf-8 -*-

"""MonFlow Detector.
This class is responsible for detecting the objects in a given image"""

# Basic Imports
import cv2
import os
import logging
from collections import defaultdict

# Yolo Imports
from models import *
from utils.datasets import *
from utils.utils import *


class Detector:
    """Class that wraps the Yolo Detector for MonFlow.
    The code is heavily inspired by detect.py of the original implementation."""

    def __init__(self, cfg, names, weights, imgsz, device,
                 view_img=False, conf_thres=0.3, iou_thres=0.6):
        # PyTorch Backend
        self.device = torch_utils.select_device(device)

        # DarkNet Model
        self.model = Darknet(cfg, imgsz)

        # Load Model's Weights
        if weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, weights)

        self.model.to(self.device).eval()

        # Init Model
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)
        self.model(img.float())

        # Save class names and colors
        self.names = load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Show classified image?
        self.view_img = view_img

        # Save Image Size
        self.imgsz = imgsz

        # Save Thresholds
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Logger
        self.logger = logging.getLogger("monflow.detector")

    def load_img(self, imgpath):
        rawimg = cv2.imread(imgpath)  # BGR in HxWxC

        # Padded resize
        img = letterbox(rawimg, new_shape=self.imgsz)[0]

        # Convert
        # ::-1 invert the img channels (BGR to RGB)
        # transpose(2, 0, 1) transform the image HxWxC to CxHxW
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Save Resize
        cv2.imwrite(imgpath + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return rawimg, img

    def preprocess_img(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def detect(self, imgpath):
        self.logger.info("Running detection on the following image: " + imgpath)

        # Load and Preprocess IMG
        rawimg, img = self.load_img(imgpath)
        img = self.preprocess_img(img)

        # Predict
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, multi_label=False)

        # Process Predictions
        output = defaultdict(list)
        det = pred[0]

        if det is not None and len(det):
            # Rescale boxes from img to rawimg size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], rawimg.shape).round()
            s = ""

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

            # Detected Objects
            self.logger.info(s)

            # Process each prediction
            for *xyxy, conf, cls in det:
                output[int(cls.item())].append([xyxy, conf])
                if self.view_img:  # Add bbox to image
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, rawimg, label=label, color=self.colors[int(cls)])

            if self.view_img:
                cv2.imshow(imgpath, rawimg)
                if cv2.waitKey(0) == ord('q'):  # q to quit
                    raise StopIteration

        return output

