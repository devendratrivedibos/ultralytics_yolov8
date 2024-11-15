import numpy as np
from ultralytics import YOLO
import cv2
model = YOLO(r'D:/Devendra_Files/ultralytics_yolov8/medianWidth/31oct3/weights/best.pt')

if __name__ == '__main__':
    source = r'Z:\SA_DATA_2024\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-3\pcams\pcam-0000038-RL.jpg'
    source = cv2.imread(source)
    source[1200:, :] = [0, 0, 0]
    # source[end_height:, :] = [0, 0, 0]
    results_segm = model(source, imgsz=640, conf=0.7, device=0)
    annotated_frame = results_segm[0].plot(boxes=True)
    # for r in results:
    #     next(results)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    for result_seg in results_segm:
        if result_seg.masks is not None:
            for mask, box in zip(result_seg.masks, result_seg.boxes):
                class_id = int(box.cls[0])  # Access to the class ID
                confidence = box.conf[0]  # Access to the confidence
                class_name = result_seg.names[class_id]  # Class name based on the ID
                points = mask.xy.copy()
                points = [np.asarray(point) for point in points]
                points = [np.round(point).astype(np.int32) for point in points]
                for contour in points:
                    convexHull = cv2.convexHull(contour)
                    # cv2.drawContours(annotated_frame, [convexHull], -1, (255, 255, 255), 3)

                rect = cv2.minAreaRect(np.concatenate(points, axis=0))
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                center_point = np.mean(box, axis=0, dtype=np.intp)
                # annotated_frame = cv2.drawContours(annotated_frame.copy(), [box], 0, (0, 255, 0), 4)
                x_boxC, y_boxC = center_point
                target_y = 1200
                threshold = 20  # Adjust as needed
                points = points[0]
                near_indices = np.where((points[:, 1] >= target_y - threshold) & (points[:, 1] <= target_y + threshold))
                near_x = points[near_indices, 0]
                lowest_x = np.min(near_x)
                highest_x = np.max(near_x)
                point1 = (lowest_x, points[np.where(points[:, 0] == lowest_x)[0][0], 1])  # Lowest point
                # point2 = (highest_x, [np.where(points[:, 0] == highest_x)[0][0], 1])  # Highest point
                point2 = (highest_x, points[np.where(points[:, 0] == highest_x)[0][0], 1])  # Highest point
                print(point1)
                print(point2)
                cv2.line(annotated_frame, point1, point2, (255, 255, 255), thickness=4)  # Blue line
                cv2.circle(annotated_frame, point1, 5, (0, 255, 0), -1)  # Green circle for lowest
                cv2.circle(annotated_frame, point2, 5, (0, 0, 255), -1)  # Red circle for highest
    cv2.imshow('img', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()