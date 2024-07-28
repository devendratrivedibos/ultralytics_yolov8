from ultralytics import YOLO
from ultralytics import settings
#import supervision as sv
import numpy as np
# View all settings
print(settings)
import os
import glob
import pdb
# model_cls = YOLO(r'cls/asp_conc5/weights/best.pt')
model_cls = YOLO(r'cls/asp_conc3/weights/best.pt')
            

if __name__ == '__main__':

    # source = r'E:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/79180851-2989-4c49-bb65-022cb0346691/ref'
    source = r'F:/90834bf3-5a81-4f57-9bab-bfe2d4bcb11c/45836e96-a328-4b20-a45f-c8ab8706d8f6/ref/a'
    results = model_cls(source,imgsz=320, conf=0.4,device =0, save=True , name='45836e96')  # generator of Results objects
    for r in results:
        print(r.probs.top1 ,   r.probs.top1conf)

        # next(results)


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