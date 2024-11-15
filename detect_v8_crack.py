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
model_seg = YOLO('E:/Devendra_Files/ultralytics-main/ultralytics-main/cracks/train6/weights/best.pt')           


if __name__ == '__main__':
   
    source = r'K:/091a0e87-260f-455e-b10c-7376a65676c2/0da45699-2e95-4f3a-9388-e34a8824708c/r3_16/'
    results = model_seg(source,imgsz=640, conf=.3,device =0, save=True)  # generator of Results objects
    for r in results:
        next(results)