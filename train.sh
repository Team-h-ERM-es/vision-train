#!/bin/bash

echo "Training YOLO..."
yolo task=detect mode=train model=yolov8s.pt data=./data.yaml epochs=100 imgsz=640 batch=16
