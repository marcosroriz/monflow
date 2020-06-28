#!/bin/env python
# -*- coding: utf-8 -*-

"""MonFlow Command Line Interface"""
import click
import csv
import glob
import logging
import os
from collections import defaultdict

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
@click.option('--benchmark', is_flag=True, help='benchmark')
def main(cfg, names, weights, source, output, img_size, conf_thres, iou_thres, 
         device, view_img, save_txt, benchmark):
    FORMAT = '[%(levelname)s] [%(name)s] %(funcName)s(): %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    
    logger = logging.getLogger("monflow.cli")

    # Are we going to process a file or a directory?
    # Check if its a directory
    if os.path.isdir(source):
        files = glob.glob(os.path.join(source, '*.jpg'))
    else: # is a single file
        files = [source]

    # Create Detector
    det = Detector(cfg, names, weights, img_size, device, view_img, conf_thres, iou_thres)

    # Process files
    total = len(files)

    # Benchmark output dictionary
    benchdict = defaultdict(int)

    for i in range(len(files)):
        logger.info("Processing Image " + str(i + 1) + "/" + str(total))

        if not benchmark:
            out = det.detect(files[i])
            logger.info("Pedestrians detected: " + str(len(out[0])))
        else:
            out = det.benchmark(files[i])
            logger.info("Pedestrians detected: " + str(out[0]))

            # Save to our benchmark dict
            _, tail = os.path.split(files[i])
            analyzedfile = tail.split("_")[0]

            benchdict[analyzedfile] = benchdict[analyzedfile] + out[0] # Pedestrians

    # Benchmark output
    benchoutput = os.path.join(output, "benchmark-" + str(conf_thres) + "-" + str(iou_thres) + ".csv")
    with open(benchoutput, 'w', newline='') as outputfile:
        csvwriter = csv.writer(outputfile, delimiter=',')
        csvwriter.writerow(["file","pedestrians"])

        for key in benchdict:
            csvwriter.writerow([key, benchdict[key]])

        outputfile.flush()

if __name__ == "__main__":
    main()
