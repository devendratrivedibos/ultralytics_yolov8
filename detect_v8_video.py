from ultralytics import YOLO
import cv2
import time
import numpy as np
import csv
import pandas as pd
# from gps_conversion import GPS_Cord
import pdb
import warnings
warnings.filterwarnings("ignore")

model_det = YOLO(r'roadFurniture/16mar/weights/best.pt')
model_seg = YOLO(r'segmentation/7mar/weights/best.pt')
video_path = r"E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/HYDRABAD-BANGALORE_ROW_LHS.mp4"
model_seg = YOLO(r'shoulderWidth/21mar/weights/best.pt')
if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    output_video_name = "Inner_demo_trial_NOIDA_ROW_0.mp4"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total_frames' , total_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
    cv2.namedWindow("Road Furniture Detection", cv2.WINDOW_NORMAL)
    frameN = 0
    while cap.isOpened():
        success, frame = cap.read()    
        if success:
            mask_frame = frame.copy()
            height, width = frame.shape[:2]
            start_col = 0
            # end_col = width // 3
            end_col = width - (width // 3)
            # mask_frame[:, start_col:end_col] = [0, 0, 0]                 inner
            mask_frame[:,end_col:width] = [0, 0, 0]        ####outerrrr
             
            curr_frame =0
            startTime = time.time()
            startTimeSegment = time.time()
            results_segm = model_seg(mask_frame, conf=0.4, iou=0.47, imgsz=640, device=0,show_boxes = False)
            annotated_frame = results_segm[0].plot(boxes = False)
            # annotated_frame[:, start_col:end_col] =frame[:, start_col:end_col]  # RGB         values for black
            annotated_frame[:,end_col:width] =frame[:,end_col:width]
            for result_seg in results_segm:
                if result_seg.masks is not None:
                    # annotated_frame[:, start_col:end_col] =frame[:, start_col:end_col]  # RGB values for black
                    annotated_frame[:,end_col:width] =frame[:,end_col:width]
                    for mask, box in zip(result_seg.masks, result_seg.boxes):
                        class_id = int(box.cls[0])  # Access to the class ID
                        confidence = float(box.conf[0].cpu())    # Access to the confidence
                        class_name = result_seg.names[class_id]  # Class name based on the ID
                        label = f"{class_name}: {confidence:.2f}"
                        box_coords_xyxy = box.xyxy.cpu().numpy()         ###CORDINATES BOX
                        box_coords_xywh = box.xywh.cpu().numpy()
                        xs1, ys1 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1])  
                        xsc, ysc = int(box_coords_xywh[0][0]), int(box_coords_xywh[0][1])
                        box_center_label =f'Centre pixel cordinates are  {str(xsc)}  {str(ysc)}'
                        points = mask.xy.copy()
                        points = [np.asarray(point) for point in points]
                        points = [np.round(point).astype(np.int32) for point in points]                       
                        for contour in points:
                            convexHull = cv2.convexHull(contour)
                            cv2.drawContours(annotated_frame, [convexHull], -1, (255, 255, 255), 3)   

                        rect = cv2.minAreaRect(np.concatenate(points, axis=0))
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        xd1 , yd1 = (box[0][0],box[0][1])
                        xd2 , yd2= (box[1][0],box[1][1])
                        xd3 , yd3= (box[2][0],box[2][1])
                        xd4 , yd4 = (box[3][0],box[3][1])
                        center_point = np.mean(box, axis=0, dtype=np.int0)
                        x_boxC , y_boxC = center_point

                        if class_id == 0:
                            cv2.putText(annotated_frame, label, (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                            cv2.putText(annotated_frame,box_center_label, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                        elif class_id == 1:   
                            cv2.putText(annotated_frame, label, (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                            cv2.putText(annotated_frame, box_center_label, (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)  
                        elif class_id == 2:   
                            cv2.putText(annotated_frame, label, (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                            cv2.putText(annotated_frame, box_center_label, (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)      
            endTimeSegment = time.time()
            segmentTime = endTimeSegment-startTimeSegment
            print("segmentTime processing time: {:.4f} seconds".format(segmentTime))

            startTimeDetect = time.time()
            results_det = model_det(mask_frame, conf=0.40, imgsz=1024, device=0)
            for result_det in results_det:
                for box in result_det.boxes:
                    xd,yd,wd,hd = box.xywh.cpu().numpy()[0]
                    xd,yd,wd,hd = int(xd),int(yd),int(wd),int(hd)
                    xd1,yd1,xd4,yd4 = (box.xyxy.cpu().numpy()[0])
                    xd1,yd1,xd4,yd4 = int(xd1),int(yd1),int(xd4),int(yd4)
                    xd3 = xd1
                    xd2 = xd4
                    yd3 = yd4
                    yd2 = yd1

                    class_id_det = int(box.cls.cpu().numpy())
                    class_name_det = result_det.names[int(box.cls)]
                    confidence_det = float(box.conf.cpu())
                    label_det = f"{class_name_det}: {confidence_det:.2f}"
                    label_centre_coord = f"{class_name_det}: {confidence_det:.2f}"
                    cv2.putText(annotated_frame, label_det, (xd1, yd1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
                    cv2.putText(annotated_frame, box_center_label, (1500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

            endTimeDetect = time.time()
            detectTime = endTimeDetect-startTimeDetect
            print("DETECT processing time: {:.4f} seconds".format(detectTime))
            results_segm = None
            results_det = None
            endTime = time.time()
            # output_video.write(annotated_frame)
            
            frameProcessTime = endTime - startTime
            print("Frame processing time: {:.4f} seconds".format(frameProcessTime))
            cv2.imshow("Road Furniture Detection", annotated_frame)
            frameN = frameN + 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # output_video.release()
    cap.release()
    cv2.destroyAllWindows()


    