import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def main():

    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Video PATH
    # video_path = "D:/Github/YOLO-yolov5-and-yolov8-Face-Detection/People.mp4"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


    # Model Path
    PATH = "D:/Github\YOLO-yolov5-and-yolov8-Face-Detection/yolov8_files/detect/train/weights/best.pt"
    # Load a model
    model = YOLO(PATH)  # pretrained YOLOv8n model

    
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )



    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, agnostic_nms=True)

            # Visualize the results on the frame
            frame = results[0].plot()
            # detections = sv.Detections.from_yolov8(results[0])
            

        #     frame = box_annotator.annotate(
        #     scene=frame, 
        #     detections=detections, 
        #     # labels=labels
        # )

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()