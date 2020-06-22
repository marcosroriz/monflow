#!/bin/env python
# -*- coding: utf-8 -*-

"""MonFlow Command Line Interface"""
import click
import logging

# MonFlow Imports
from detector import Detector


@click.command()
@click.option('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
@click.option('--names', type=str, default='data/coco.names', help='*.names path')
@click.option('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
@click.option('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
@click.option('--output', type=str, default='output', help='output folder')  # output folder
@click.option('--img-size', type=int, default=512, help='inference size (pixels)')
@click.option('--conf-thres', type=float, default=0.3, help='object confidence threshold')
@click.option('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
@click.option('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
@click.option('--view-img', is_flag=True, help='show classified image')
@click.option('--save-txt', is_flag=True, help='save results to *.txt')
def main(cfg, names, weights, source, output, img_size, conf_thres, iou_thres, 
         device, view_img, save_txt):
    logging.basicConfig(level=logging.INFO)

    det = Detector(cfg, names, weights, img_size, device, view_img)
    out = det.detect(source)
    print(out)


if __name__ == "__main__":
    main()
