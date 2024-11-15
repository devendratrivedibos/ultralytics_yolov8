from ultralytics import YOLO
from ultralytics import settings
#import supervision as sv
import numpy as np
# View all settings
print(settings)
import os


# model = YOLO('22feb_RF/train/weights/best.pt')
model = YOLO('segmentation/pot_patch/weights/best.pt')
# model_seg = YOLO('24Feb_seg/train/weights/best.pt')                

if __name__ == '__main__':
   
    source = r'Y:/89b23187-36ed-44e8-be9a-8a934988cc2b/a908b650-3bd8-4fa7-b67a-210dc64e6ddd/range_images/YOLODataset/images/test/'
    results = model(source,imgsz=640, conf=0.4,save_txt = True,save=True,device =0, stream =True)  # generator of Results objects    
    # results = model(source,imgsz=640, conf=0.4,iou=0.45,save_txt = True,save=True,device =0, stream =True)  # generator of Results objects
    for r in results:
        next(results)
    for result in results:
        if result.masks is not None:
            for mask in result.masks:
                for box in result.boxes:
                    class_id = box.cls.cpu().numpy()
                    label = result.names[int(box.cls)]
                    points = mask.xy
                    points = np.round(points).astype(np.int32)
                    # points_for_cv2 = points[:, np.newaxis, :]
                    for contour in points:
                        convexHull = cv2.convexHull(contour)
                    cv2.drawContours(annotated_frame, [convexHull], -1, (255, 0, 0), 4)

                # x, y, w, h = cv2.boundingRect(points)
                # annotated_frame = cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                rect = cv2.minAreaRect(points)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                                