from ultralytics import YOLO
import cv2
import time
import numpy as np
model = YOLO(r'22feb_RF/train/weights/best.pt')
model = YOLO(r'24Feb_seg/train2/weights/best.pt')
source = 'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/0f2bd68e-ddd9-4345-905f-563a573cb181/data_set/'
video_path = "W:/Devendra_Files/Snippets/projectLR.mp4"
# video_path = "K:/0031c4a0-91e0-4f6b-ab86-c9ed45c59384/SALEM-ULUNDARPET_ROW_1.mp4"

if __name__ == '__main__':

    cap = cv2.VideoCapture(video_path)
    output_video_name = "Inventory_Tracking_Side.mp4"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    #output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
    cv2.namedWindow("Road Furniture Detection", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # results = model.track(frame, persist=True,conf=0.60, iou=0.70, imgsz=640, device=0)
            results = model.predict(frame, conf=0.3, iou=0.4, imgsz=640, device=0, show_boxes = False)
            # print(results)
            # annotated_frame = results[0].plot()
            annotated_frame = results[0].plot(boxes=False)
            # cv2.imshow("Road Furniture Detection", annotated_frame)
            #output_video.write(annotated_frame)


            for result in results:
                print (result)

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
                        
                        annotated_frame = cv2.drawContours(annotated_frame.copy(), [box], 0, (0, 255, 0), 4)

                cv2.imshow("Road Furniture Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # output_video.release()
    cap.release()
    cv2.destroyAllWindows()

