# MonFlow

## Introduction

MonFlow aims to **mon**itor traffic **flow** through surveillance cameras. 
It provides a microservice that receive a static image as input and outputs the number of pedestrians and vehicles that are considered as moving on the picture. Hence, it discards objects that are still, *e.g.*, cars that are parked.

By providing this service, we aim to monitor the influence of the city policy, on COVID-19, in the traffic flow.

## Implementation Details

MonFlow is built on top of the excellent [UltraLytics’s yolo v3](https://github.com/ultralytics/yolov3) implementation. Precisely, we customized the original code to train on the surveillance camera images. Furthermore, we used OpenCV to preprocess the incoming images.

As a microservice, it provides a command line interface (CLI) and HTTP interface to interconnect with the reaming system’s components. 


## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.5`. To install run:
```bash
$ pip install -U -r requirements.txt
```

## Training

**Start Training:** 

```bash
$ !python3 train.py --data data/coco_1cls.data --cfg cfg/yolov3-spp.cfg --weights weights/yolov3-spp-ultralytics.pt
```

**Resume Training:**

```bash
$ !python3 train.py --data data/coco_1cls.data --cfg cfg/yolov3-spp.cfg --weights weights/last.pt --epochs 500
```

## Inference

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8`

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067835-51d5b500-cc2f-11e9-982e-843f7f9a6ea2.jpg" width="500">


## Pretrained Checkpoints

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)


## Darknet Conversion

```bash
$ git clone https://github.com/ultralytics/yolov3 && cd yolov3

# convert darknet cfg/weights to pytorch model
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'weights/yolov3-spp.pt'

# convert cfg/pytorch model to darknet weights
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.pt')"
Success: converted 'weights/yolov3-spp.pt' to 'weights/yolov3-spp.weights'
```
