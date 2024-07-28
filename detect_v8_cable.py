from ultralytics import YOLO
from ultralytics import settings
#import supervision as sv
import numpy as np
# View all settings
# print(settings)
import os
import glob
import pdb
import cv2
# model_seg = YOLO('segmentation/pot_patch7/weights/best.pt')           
model_seg = YOLO('E:/Devendra_Files/ultralytics-main/ultralytics-main/cable/train/weights/best.pt')           


if __name__ == '__main__':
   
    source = r'D:/Cable Inspection/New folder/train_data_sample/'
    results = model_seg(source,imgsz=640, conf=.3,device =0, save=True)  # generator of Results objects
    for r in results:
        next(results)