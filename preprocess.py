#!/bin/env python
# -*- coding: utf-8 -*-

"""MonFlow Preprocessor.
This class is responsible for preprocess the incoming image"""

# Basic Imports
import base64
import cv2
import os
import logging
from collections import defaultdict


class Preprocessor:
    """Class that provides a suite of preprocessing functions for handling the incoming image."""

    def __init__(self):
        # Logger
        self.logger = logging.getLogger("monflow.preprocessor")

    def file_to_base64(self, imgfile):
        with open(imgfile, "rb") as file:
            imgstr = base64.b64encode(file.read()).decode('utf-8')

        return imgstr


if __name__ == '__main__':
    prep = Preprocessor()
    print(prep.file_to_base64("data/samples/grande.jpg")[:100])

    img = cv2.imread("data/samples/grande.jpg")
    height, width = img.shape[:2]
    ym, xm = int(height / 2), int(width / 2)
    i0 = img[:ym, :xm, :]
    i1 = img[:ym, xm:, :]
    i2 = img[ym:, :xm, :]
    i3 = img[ym:, xm:, :]

    print(img.shape)
    print("topleft", i0.shape)
    print("topright", i1.shape)
    print("bottomleft", i2.shape)
    print("bottomright", i3.shape)

    cv2.imshow("complete", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("top left", i0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("top right", i1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("bottom left", i2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("bottom right", i3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
