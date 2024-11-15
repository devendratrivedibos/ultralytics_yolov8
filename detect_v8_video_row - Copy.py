from ultralytics import YOLO
import cv2
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
model_cls = YOLO(r'cls/asp_conc5/weights/best.pt')
import pandas as pd
video_path = "K:/survey_data/2d46fac9-fd1d-47e5-ad8d-c087a671c2d2/Akkalkot_SURFACE.mp4"

def roadClassification(img):
    result_classification =  model_cls(img, conf=0.6, iou=0.5, imgsz=640, device=0)
    for result_cls in result_classification:
        class_id = result_cls.probs.top1
        class_conf = result_cls.probs.top1conf.cpu()
        if class_id == 1:
            cv2.putText(img, str('Concrete') + str(class_conf), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
        if class_id == 0:
            cv2.putText(img, str('Asphalt') + str (class_conf), (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
    print(class_id) 
    return class_id

img = cv2.imread("K:/survey_data/2d46fac9-fd1d-47e5-ad8d-c087a671c2d2/1b6100b6-548b-4dd8-9d3f-5a708da3961d/ref/ref_7.jpg")
if __name__ == '__main__':
    roadClassification(img)