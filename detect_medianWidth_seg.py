from ultralytics import YOLO
import numpy as np
import cv2
import pdb
model_seg = YOLO('medianWidth/31oct3/weights/best.pt')

if __name__ == '__main__':
    sourceseg = r'Z:/SA_DATA_2024/LALGANJ-HANUMANA_2024-10-05_10-23-09/SECTION-4\pcams/pcam-0000007-RL.jpg'
    sourceseg = cv2.imread(sourceseg)
    height, width = sourceseg.shape[:2]
    end_col = width // 2
    # sourceseg[1600:,] = [0, 15, 1]  # RGB values for black
    results_median_segm = model_seg(sourceseg, imgsz=640, conf=0.3,iou= 0.3,device =0,show_boxes = False)  # generator of Results objects
    annotated_frame = results_median_segm[0].plot(boxes=False)
    for result_seg in results_median_segm:
        if result_seg.masks is not None:
            if len(result_seg.masks) != len(result_seg.boxes):
                print("Warning: Mismatched lengths of masks and boxes.")
                continue  # Skip this segment if lengths don't match
            for mask, box in zip(result_seg.masks, result_seg.boxes):
                points = mask.xy.copy()
                points = [np.asarray(point) for point in points]
                points = [np.round(point).astype(np.int32) for point in points]
                target_y = 1600
                threshold = 10
                points = points[0]
                near_indices = np.where(
                    (points[:, 1] >= target_y - threshold) & (points[:, 1] <= target_y + threshold))
                near_x = points[near_indices, 0]

                if near_x.size > 0:
                    lowest_x = np.min(near_x)
                    highest_x = np.max(near_x)

                else:
                    print("No points found near target y.")
        try:
            point1 = (lowest_x, points[np.where(points[:, 0] == lowest_x)[0][0], 1])  # Lowest point
            point2 = (highest_x, points[np.where(points[:, 0] == highest_x)[0][0], 1])  # Highest point
            print("Point 1:", point1)
            print("Point 2:", point2)
            median_width = point1[0] - point2[0]
            point1 = (lowest_x, points[np.where(points[:, 0] == lowest_x)[0][0], 1]-10)
            point2 = (highest_x, points[np.where(points[:, 0] == highest_x)[0][0], 1]-10)
            #median_width = Sheet.calcLength(point1, point2)
            cv2.line(annotated_frame, point1, point2, (255, 0, 255), thickness=2)  # Blue line
            cv2.circle(annotated_frame, point1, 5, (0, 255, 0), -1)  # Green circle for lowest
            cv2.circle(annotated_frame, point2, 5, (0, 0, 255), -1)  # Red circle for highest
            cv2.putText(annotated_frame, f"Median Distance is {median_width}", (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 4)

        except Exception as e:
            print('_______________s', e)
            x1_diff = [0]

    cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    cv2.imshow('im', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
