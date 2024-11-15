from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO(r'D:/Devendra_Files/ultralytics_yolov8/medianWidth/31oct3/weights/best.pt')
video_path = r"Z:\SA_DATA_2024\LALGANJ-HANUMANA_2024-10-05_10-23-09\SECTION-4/LALGANJ-HANUMANA_ROW_SECTION-4.mp4"

if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    print(fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cv2.namedWindow("Road Furniture Detection", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        end_height = int(frame_height - frame_height/4)
        frame[end_height:, :] = [255, 0, 0]
        if success:
            results_segm = model.predict(frame, conf=0.3, iou=0.4, imgsz=640, device=0, show_boxes = False)
            annotated_frame = results_segm[0].plot(boxes=True)

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
                        target_y = 1540
                        threshold = 20  # Adjust as needed
                        points = points[0]
                        near_indices = np.where(
                            (points[:, 1] >= target_y - threshold) & (points[:, 1] <= target_y + threshold))
                        near_x = points[near_indices, 0]

                        if near_x.size > 0:
                            lowest_x = np.min(near_x)
                            highest_x = np.max(near_x)

                            point1 = (lowest_x, points[np.where(points[:, 0] == lowest_x)[0][0], 1])  # Lowest point
                            point2 = (highest_x, points[np.where(points[:, 0] == highest_x)[0][0], 1])  # Highest point

                            print("Point 1:", point1)
                            print("Point 2:", point2)

                            # Draw the line and circles on the annotated frame
                            cv2.line(annotated_frame, point1, point2, (255, 0, 0), thickness=2)  # Blue line
                            cv2.circle(annotated_frame, point1, 5, (0, 255, 0), -1)  # Green circle for lowest
                            cv2.circle(annotated_frame, point2, 5, (0, 0, 255), -1)  # Red circle for highest
                        else:
                            print("No points found near target y.")

            cv2.imshow("Road Furniture Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

