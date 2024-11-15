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
model_seg = YOLO('segmentation/pot_patch4/weights/best.pt')           


if __name__ == '__main__':
    poth_area_list = []
    poth_widt_list = []
    poth_heith_list = []
    num_of_poth_list = 0

    patch_area_list = []
    patch_widt_list = []
    patch_heith_list = []
    num_of_patch_list = 0
    source = "E:/Ashutosh B Stuff/all_annotations/YOLODataset/images/new_test"
    # annotated_frame = source.copy()
    source = "F:/ee74a59d-0345-480c-9853-18e3ef55d780/a5c63eb8-49e1-4932-b6ef-d5a65927c5ad/testimages/"
    results_segm = model_seg(source,classes=[0,1],imgsz=1024, conf=0.3,device=0, save=True)  # generator of Results objects
    for result_seg in results_segm:
    
        poth_area_list = []
        poth_widt_list = []
        poth_heith_list = []
        num_of_poth_list = 0

        patch_area_list = []
        patch_widt_list = []
        patch_heith_list = []
        num_of_patch_list = 0

        if result_seg.masks is not None:
                # annotated_frame[:, start_col:end_col] =frame[:, start_col:end_col]  # RGB values for black
                for mask, box in zip(result_seg.masks, result_seg.boxes):
                    class_id = int(box.cls[0])  # Access to the class ID
                    confidence = float(box.conf[0].cpu())    # Access to the confidence
                    class_name = result_seg.names[class_id]  # Class name based on the ID
                    label = f"{class_name}: {confidence:.2f}"
                    box_coords_xyxy = box.xyxy.cpu().numpy()         ###CORDINATES BOX
                    box_coords_xywh = box.xywh.cpu().numpy()
                    xs1, ys1 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1])  
                    xsc, ysc = int(box_coords_xywh[0][0]), int(box_coords_xywh[0][1])
                    xyw, xyh = int(box_coords_xywh[0][2]), int(box_coords_xywh[0][3])

                    box_center_label =f'Centre pixel cordinates are  {str(xsc)}  {str(ysc)}'
                    widt_n_ieit = f'widt_n_ieit pixel cordinates are  {str(xyw)}  {str(xyh)}'
                    print(f"widt_n_ieit of the segmented mask: {widt_n_ieit}  pixels")
                    points = mask.xy.copy()
                    points = [np.asarray(point) for point in points]
                    points = [np.round(point).astype(np.int32) for point in points]
                    # print(points)
                    for contour in points:
                        convexHull = cv2.convexHull(contour)
                        cv2.drawContours(annotated_frame, [convexHull], -1, (255, 255, 255), 3)   
                        area = cv2.contourArea(convexHull)
                        print(f"Area of the segmented mask: {area} square pixels")
                        area_label = f"Area of the segmented mask: {area} square pixels"
                    rect = cv2.minAreaRect(np.concatenate(points, axis=0))
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    xd1 , yd1 = (box[0][0],box[0][1])
                    xd2 , yd2= (box[1][0],box[1][1])
                    xd3 , yd3= (box[2][0],box[2][1])
                    xd4 , yd4 = (box[3][0],box[3][1])
                    center_point = np.mean(box, axis=0, dtype=np.intp)
                    # annotated_frame = cv2.drawContours(annotated_frame.copy(), [box], 0, (0, 255, 0), 4)
                    x_boxC , y_boxC = center_point

                    box_center_label =f'{class_name} Centre pixel cordinates are  {str(x_boxC)}  {str(y_boxC)}'

                    if class_id == 0:
                        poth_area_list.append(area)
                        poth_widt_list.append(xyw)
                        poth_heith_list.append(xyh)
                        num_of_poth_list = num_of_poth_list+1
                        cv2.putText(annotated_frame, label +str(area_label) +str(widt_n_ieit), (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                        cv2.putText(annotated_frame,box_center_label, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

                    elif class_id == 1:
                        patch_area_list.append(area)
                        patch_widt_list.append(xyw)
                        patch_heith_list.append(xyh)
                        num_of_patch_list = num_of_patch_list+1

                        cv2.putText(annotated_frame, label +str(area_label) +str(widt_n_ieit), (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                        # cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                        cv2.putText(annotated_frame, box_center_label, (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)  
                    elif class_id == 2:   
                        cv2.putText(annotated_frame, label + str(area_label), (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                        # cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                        cv2.putText(annotated_frame, box_center_label, (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4) 

                # cv2.imwrite("E:/Devendra_Files/ultralytics-main/ultralytics-main/img.png" , annotated_frame)
        print(num_of_poth_list, poth_area_list, poth_widt_list, poth_heith_list, num_of_patch_list, patch_area_list, patch_widt_list, patch_heith_list)
