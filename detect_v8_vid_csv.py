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

if __name__ == '__main__':
    frame = r'E:\LUCKNOW-AYODHYA_2024-08-30_09-10-40\SECTION-1\ref_original'
    results = model_cls.predict(frame, conf=0.3, iou=0.4, imgsz=320, device=0, save_txt=True)
    for r in results:
        print(r.probs.top1 ,   r.probs.top1conf)
        next(results)

