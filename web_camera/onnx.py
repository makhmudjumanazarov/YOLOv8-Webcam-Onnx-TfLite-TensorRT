import matplotlib.pyplot as plt
import onnxruntime as ort
import numpy as np
import cv2
from letterbox import letterbox

session = ort.SessionOptions()
session = ort.InferenceSession('yolov8_files/detect/train/weights/yolov8_changed.onnx', providers=['CPUExecutionProvider','CUDAExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(0)

if (cap.isOpened()== False):
    print("Error opening video file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        image, ratio, dwdh = letterbox(frame, auto=False)
        
        # frame = cv2.resize(frame, (640, 640))
        image = image.transpose(2,0,1) / 255.
        image = image[np.newaxis, :, :, :].astype(np.float32)
        output_data = session.run([output_name], {input_name: image})[0][0]
        output_data = output_data[:, output_data[1].argsort()]
        output_data = output_data[:, output_data[0].argsort()]

        for (x0, y0, x1, y1, scores) in output_data[:,output_data[4]>0.1].T[::10]:
            if scores > 0.3:
                x0 -= dwdh[0]
                y0 -= dwdh[1]
                box = np.array([x0-x1/2, y0-y1/2, x0+x1/2, y0+y1/2])
                box /= ratio
                box = np.round(box).astype(np.int32).tolist()

                score = round(float(scores), 2)

                cv2.rectangle(frame, box[:2], box[2:], (0,255,0), 2)
                cv2.putText(frame, str(score), (box[0], box[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [0, 255, 255], thickness=2)
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()