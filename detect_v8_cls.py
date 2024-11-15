from ultralytics import YOLO
from ultralytics import settings
#import supervision as sv
import numpy as np
# View all settings
print(settings)
import os
import glob
import pdb
import cv2
model_cls = YOLO(r'cls/asp_conc5/weights/best.pt')
import csv


if __name__ == '__main__':

    source = r'F:/90834bf3-5a81-4f57-9bab-bfe2d4bcb11c/45836e96-a328-4b20-a45f-c8ab8706d8f6/ref/a'
    video_path = r"Z:\SA_DATA_2024\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-1/LALGANJ-HANUMANA_SURFACE_SECTION-1.mp4"
    cap = cv2.VideoCapture(video_path)
    # cv2.namedWindow("IMG", cv2.WINDOW_NORMAL)
    with open('SECTION-4.csv', mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow([['Section_ID', 'Region_index', 'Road type']])
        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model_cls.predict(frame, conf=0.3, iou=0.4, imgsz=320, device=0, show=True)
                for r in results:
                    top1_prob = r.probs.top1
                    csv_writer.writerow(['SECTION-4',frame_number, top1_prob])

                    # # Display the frame
                    # cv2.imshow("IMG", frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                frame_number += 1  # Increment frame counter
            else:
                break

    # Release video capture
    cap.release()

