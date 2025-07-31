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

-	TensorFlow: Machine learning framework for model development.

-	TensorFlow Object Detection API: For building and deploying the detection model.

-	Python: Core programming language for implementation.

-	Computer Vision: Techniques for image processing and gesture recognition.

-	Deep Learning: Neural networks for model training.

-	Transfer Learning: Pre-trained models to improve performance.

-	Labelimg: For dataset preparation and annotation.


## Prerequisites

- Python 3.8+
- Anaconda
- pip
- TensorFlow 2.10
- TensorFlow Object Detection API
- OpenCV
- LabelImg (for dataset labeling)
- SSD MobileNet (pre-trained base model)
- Protoc 3.20.3
- Protobuf 3.20.3

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
└── README.md                           # Readme file
```

## Installation

1. Clone the repository

        git clone https://github.com/adhyxn/Real-Time-Sign-Language-Detection.git
    
        cd real-time-sign-language-detection

2. Install prerequisites and dependencies

    • Install Anaconda to install tensorflow further and create a custom python environment.
    
    • Install Labelimg from github.

    https://github.com/HumanSignal/labelImg.git
    
    • Install tensorflow 2.10
    
       pip install tensorflow==2.10.0
    
    • Install protoc 3.20.3
    
    • Install protobuf 3.20.3
    
    • Other dependencies
    
        pip install opencv-python numpy pillow

4. Install TensorFlow Object Detection API (Follow these steps to set up the Object Detection API):

       git clone https://github.com/tensorflow/models.git
   
       cd models/research
   
       protoc object_detection/protos/\*.proto --python_out=.
   
       cp object_detection/packages/tf2/setup.py
   
       pip install .
   
       cd ../..

5. Prepare the Dataset

    Use the project.ipynb file to start the webcam and collect and label sign language images using LabelImg.
   
    Organize the dataset into training and testing sets.
   
    Convert annotations to TFRecord format (refer to TensorFlow Object Detection API documentation).

6. Configure the Model

    Update the model configuration file (e.g., pipeline.config) with paths to the pre-trained model, dataset, and other parameters.

6. Train the Model

    Run the training script, specifying the configuration file and output directory.

        python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=10000

7. Export the Trained Model

    Export the trained model for inference.

8. Run Real-Time Detection

    Use the provided inference script to perform real-time detection with a webcam.

## Acknowledgments

- TensorFlow Object Detection API

- OpenCV

- Labelimg

- TensorFlow Model Zoo for pretrained models
