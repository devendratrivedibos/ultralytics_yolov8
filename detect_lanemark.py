from ultralytics import YOLO
import cv2
import time
import numpy as np
import csv
import pandas as pd
from gps_conversion import GPS_Cord

import warnings

model_seg = YOLO(r"D:\Devendra_Files\ultralytics_yolov8\segmentation\lanemark2\weights\best.pt")
video_path = r"Z:\SA_DATA_2024\LALGANJ-HANUMANA_2024-10-05_10-23-09/LALGANJ-HANUMANA_SURFACE_LHS_OUTER.mp4"
# video_path = r"Z:\SA_DATA_2024\LUCKNOW-RAEBARELI_2024-10-02_12-14-16/LUCKNOW-RAEBARELI_ROW_SECTION-11.mp4"
# model_seg = YOLO(r"D:\Devendra_Files\ultralytics_yolov8\shoulderWidth\7nov\weights\best.pt")

def sort_coordinate_lists(x_list, y_list):
    """
    Sorts the x and y coordinate lists based on the x values.

    Args:
        x_list (list): List of x coordinates.
        y_list (list): List of y coordinates.

    Returns:
        tuple: Sorted x and y lists.
    """
    combined = list(zip(x_list, y_list))
    combined = sorted(combined, key=lambda pair: pair[0])
    x_list, y_list = zip(*combined)
    return list(x_list), list(y_list)




if __name__ == '__main__':

    confList = []
    xd1list = []
    xd4list = []

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total_frames', total_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frameN = 0
    # cv2.namedWindow("Road Furniture outer", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        xd1list = []
        xd4list = []
        yd1list = []
        yd4list = []
        success, frame = cap.read()
        if success:
            # if frame is not None and frame.shape[1] == 400 and frame.shape[0] == 1000:
            #     frame = cv2.resize(frame, (1280, 720))
                # cv2.imwrite(image_path, resized_image)
            mask_frame = frame.copy()
            startTimeDetect = time.time()
            curr_frame = 0
            startTime = time.time()
            startTimeSegment = time.time()
            results_segm = model_seg(mask_frame, conf=0.4, iou=0.2, imgsz=640, device=0, show_boxes=True)
            annotated_frame = results_segm[0].plot(boxes=False)
            try:
                for result_seg in results_segm:
                    if result_seg.masks is not None:
                        for mask, box in zip(result_seg.masks, result_seg.boxes):
                            class_id = int(box.cls[0])  # Access to the class ID
                            confidence = float(box.conf[0].cpu())  # Access to the confidence
                            class_name = result_seg.names[class_id]  # Class name based on the ID
                            label = f"{class_name}: {confidence:.2f}"
                            box_coords_xyxy = box.xyxy.cpu().numpy()  ###CORDINATES BOX
                            box_coords_xywh = box.xywh.cpu().numpy()
                            xd1, yd1, xd4, yd4 = box.xyxy.cpu().numpy()[0]

                            xd1list.append(int(xd1))
                            xd4list.append(int(xd4))
                            yd1list.append(int(yd1))
                            yd4list.append(int(yd4))
                            points = mask.xy.copy()
                            points = [np.asarray(point) for point in points]
                            points = [np.round(point).astype(np.int32) for point in points]
                            for contour in points:
                                convexHull = cv2.convexHull(contour)
                                cv2.drawContours(annotated_frame, [convexHull], -1, (255, 255, 255), 1)
                    xd1list, yd1list = sort_coordinate_lists(xd1list, yd1list)
                    xd4list, yd4list = sort_coordinate_lists(xd4list, yd4list)

                    cv2.line(annotated_frame, (xd4list[0], yd4list[0]), (xd1list[-1], yd4list[0]), (255, 125, 125), 2)
            except Exception as e:
                print(e)

            cv2.imshow("Road Furniture outer", annotated_frame)
            frameN = frameN + 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # output_video.release()
    cap.release()
    cv2.destroyAllWindows()
