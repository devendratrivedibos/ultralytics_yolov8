from ultralytics import YOLO
import cv2
import time
import numpy as np
import pandas as pd
from gps_conversion import GPS_Cord

model = YOLO(r'D:/Devendra_Files/ultralytics_yolov8/shoulderWidth/29julyy3/weights/soulderWidt.pt')

if __name__ == '__main__':
    video_path = 'D:/ALIGARH-KANPUR_2024-09-20_14-18-46/SECTION-7/ALIGARH-KANPUR_ROW_0.mp4'
    cap = cv2.VideoCapture(video_path)
    output_video_name = "Inventory_Lane.mp4"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
    cv2.namedWindow("Road Furniture Detection", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results_det = model(frame, conf=0.50, iou=0.06 ,imgsz=640, device=0, save=False)
            annotated_frame = results_det[0].plot(boxes=False)

            cv2.imshow("Road Furniture Detection", annotated_frame)
            output_video.write(annotated_frame)            

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    output_video.release()
    cap.release()
    cv2.destroyAllWindows()

