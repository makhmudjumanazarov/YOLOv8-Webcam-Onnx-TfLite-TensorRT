import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from ultralytics import YOLO
from PIL import Image
import tensorflow as tf
import pandas as pd
import onnxruntime 

import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx
import onnxruntime as rt
from onnxruntime.datasets import get_example
import torch.onnx

import time
import supervision as sv

box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

def prediction_onnx(model_path, image):
    opt_session = onnxruntime.SessionOptions()
    opt_session.enable_mem_pattern = False
    opt_session.enable_cpu_mem_arena = False
    opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    # model_path = 'models/best.onnx'
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)

    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape

    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]

    # image = cv2.imread('data/sample_images/2.jpg')
    image_height, image_width = image.shape[:2]
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    input_height, input_width = input_shape[2:]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_width, input_height))

    # Scale input pixel value to 0 to 1
    input_image = resized / 255.0
    input_image = input_image.transpose(2,0,1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

    outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]

    predictions = np.squeeze(outputs).T
    conf_thresold = 0.3
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]

    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = predictions[:, :4]

    #rescale box
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_width, image_height, image_width, image_height])
    boxes = boxes.astype(np.int32)
#     print(boxes)

    def nms(boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def compute_iou(box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(boxes, scores, 0.0)

    CLASSES = [
    'Face'
    ]


    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    image_draw = image.copy()
    for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
        bbox = bbox.round().astype(np.int32).tolist()
#         print(bbox)
        cls_id = int(label)
        # cls = CLASSES[cls_id]
        color = (0,255,0)
        cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        cv2.putText(image_draw,
                    f'Person:{int(score*100)}', (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, [225, 0, 255],
                    thickness=2)
    return image_draw

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 0
fpss = []
prev_time = 0
curr_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width, height))
    output_img = prediction_onnx(model_path = "D:\Github\Object-detection-YOLO-Pytorch-ONNX-Tensorrt\yolov8_files\detect\train\weights\model_quant_dynamic.onnx", image = frame)
    cv2.imshow("text", output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # plt.imshow(output_img)
    # plt.show()
    
    curr_time = time.time()
    fps = (1 / (curr_time - prev_time))*2
    print("FPS: --", fps)
    fpss.append(fps)
    prev_time = curr_time
# plt.plot(fpss)

cap.release()
cv2.destroyAllWindows()
# plt.savefig('MyPlot_onnx.png')


