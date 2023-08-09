import torch
import numpy as np
import cv2
import pandas as pd 
import onnx

from matplotlib import pyplot as plt

PATH = "yolov8_files/detect/train/weights/best.pt"


# How to convert pytorch to onnx format
device = torch.device('cpu')
model = torch.load(PATH, map_location=device)['model'].float()
torch.onnx.export(model, torch.zeros((1, 3, 640, 640)), 'formats/onnx.onnx', opset_version=12)

onnx_model = onnx.load("formats/onnx.onnx")
onnx.checker.check_model(onnx_model)   
