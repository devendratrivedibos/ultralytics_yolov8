from ultralytics import YOLO
from ultralytics import settings
import cv2
#import supervision as sv
import numpy as np
# View all settings
print(settings)
import os
import torch

# # model = YOLO('22feb_RF/train/weights/best.pt')
# modelobb = YOLO('24Feb_obb/train/weights/best.pt')
model_seg = YOLO('segmentation/29apr2/weights')                
model_seg.export(format='onnx')

if __name__ == '__main__':
    # source = r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/d746fc53-8f6b-4bb1-ad95-942f1e6a3060/data_set'
    sourceseg = r'E:/Devendra_Files/ultralytics-main/ultralytics-main/datasets/data_set_seg/Metal_RcC_median_GUIDE.v2i.yolov8/valid/images'
    # sourceobb = r'W:/Devendra_Files/ultralytics-main/ultralytics-main/datasets/obb/valid/images'    
    # sourceseg = r'E:/Devendra_Files/ultralytics-main/ultralytics-main/datasets/data_set_seg/valid/images'   
    
    results = model_seg(sourceseg,imgsz=640, conf=0.4,device =0, save = True,stream=True)  # generator of Results objects 
    for result in results:
        for mask in result.masks:
            
            xy = mask.xy
            points = xy.copy()
            # Convert points to integers
            points = np.round(points).astype(np.int32)
            print(points)
            annotated_frame = result[0].plot()
            # Reshape points to the required format
            points_for_cv2 = points[:, np.newaxis, :]
            print(points_for_cv2)
            for contour in points_for_cv2:
                convexHull = cv2.convexHull(contour)
                cv2.drawContours(annotated_frame, [convexHull], -1, (255, 0, 0), 2)

    # print(results[0].masks.xy) 
    # print(len(results[0].masks.xy[0]))
    # print(results[0].boxes.xy) 
    # Check if masks are available in the result
    # if results[0].masks is not None:
    #     # Convert mask to numpy array
    #     masks = results[0].masks.cpu().numpy()

    #     # Get the first mask
    #     mask = masks[0]

    #     # Apply the mask to the image
    #     segmented_img = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=mask)

    #     # Display the segmented image
    #     cv2.imshow("Segmented Image", segmented_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


    # if(results[0].masks is not None):
    #     mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
    #     mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
    #     h2, w2, c2 = results[0].orig_img.shape
    #     mask = cv2.resize(mask_3channel, (w2, h2))
    #     hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    #     lower_black = np.array([0,0,0])
    #     upper_black = np.array([0,0,1])
    #     mask = cv2.inRange(mask, lower_black, upper_black)
    #     mask = cv2.bitwise_not(mask)
    #     masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=mask)
    #     # cv2.imshow("mask", masked)
    #     cv2.imwrite('W:/Devendra_Files/ultralytics-main/ultralytics-main/imgg.jpg' , masked)



    # for result in results:
    #     # get array results
    #     masks = result.masks.data
    #     boxes = result.boxes.data

    #     # save to file
    #     cv2.imwrite(str('W:/Devendra_Files/ultralytics-main/ultralytics-main/merged_segs.jpg'), masks.cpu().numpy())