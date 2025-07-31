# Real-Time-Sign-Language-Detection

## Overview
This project implements a real-time sign language detection system using TensorFlow and OpenCV. The system accurately predicts and interprets sign language gestures from live webcam input, leveraging a custom-trained object detection model based on TensorFlow’s Object Detection API.

## Features
- Real-time gesture detection using a webcam.
- Custom dataset of gesture images collected and labeled with Labelimg.
- Transfer learning applied to a Single Shot Detector (SSD) model for efficient object detection.
- Integration with OpenCV for live video processing.
- Achieves high accuracy and low-latency in gesture prediction.

## Tech Stack
- Python  
- TensorFlow 2.x  
- TensorFlow Object Detection API  
- OpenCV  
- LabelImg (for dataset labeling)  
- SSD MobileNet (pre-trained base model)

## Project Structure
```
real-time-sign-language-detection/
├── Tensorflow/
│   ├── workspace/
│   │   ├── annotations/
│   │   │   ├── label_map.pbtxt
│   │   │   ├── train.record
│   │   │   └── test.record
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── test/
│   │   ├── models/
│   │   │   └── my_ssd_mobnet/
│   │   │       ├── pipeline.config
│   │   │       └── ckpt-*
│   │   ├── pre-trained-models/
│   │   │   └── ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/
│   │   └── scripts/
│   │       └── generate_tfrecord.py
├── RealTimeSignLanguageDetection.ipynb  # Main notebook
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```