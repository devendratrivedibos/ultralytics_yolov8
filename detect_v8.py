from ultralytics import YOLO
from ultralytics import settings
#import supervision as sv
import numpy as np
# View all settings
print(settings)
import os
import glob

model_det = YOLO(r'roadFurniture/16mar/weights/best.pt')
# modelobb = YOLO('24Feb_obb/train/weights/best.pt')
model_seg = YOLO(r'segmentation/7mar/weights/best.pt')            


if __name__ == '__main__':

    source = r'F:/56e1fce2-4e22-4db2-978f-6038d366b759/d2294581-3d6b-4a97-a2ad-d655f556175b/data_set'
    results = model_det(source,imgsz=1024, conf=0.4,device =0,save_txt=True, name='d2294581-3d6b-4a97-a2ad-d655f556175b')  # generator of Results objects
    for r in results:
        next(results)

    # results = model.predict(selected_images,imgsz=1024, save=True,conf=0.4,save_txt = True)#, device =0)  # generator of Results objects

    # for result in results:
    #     for box in result.boxes:
    #         print('-----',box.xywhn.cpu().numpy()[0])
    #         # print('-----',box.xyxy.cpu().numpy())
    #         x,y,w,h = box.xywhn.cpu().numpy()[0]
    #         class_id = box.cls.cpu().numpy()
    #         label = result.names[int(box.cls)]
    #         confidence = float(box.conf.cpu())
    #         print("label :", label)
    #         # print("Bounding Boxes:", x1, y1, x2, y2)
    #         class_id = box.cls.cpu().numpy()    # cls, (N, 1)

    #     # annotated_frame = r[0].plot()
        #annotated_frame = r.save_text()
    #results = model(source, imgsz=1024, conf=0.4,stream = True ,save_txt = True, device = 0) 

    # # bbox = results[0].boxes.xyxy
    # cls = results.boxes.cls.cpu().numpy()    # cls, (N, 1)
    # probs = results.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    # boxes = results.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)
    # print(cls)
    # print(probs)
    # print(boxes)