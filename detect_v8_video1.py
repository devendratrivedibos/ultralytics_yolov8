from ultralytics import YOLO
import cv2
import time
import numpy as np
import pandas as pd
from gps_conversion import GPS_Cord

model = YOLO(r'segmentation/5mar2/weights/best.pt')

if __name__ == '__main__':
    video_path = 'F:/1f9c618c-9676-489b-9204-4f29d724ee22/PALANPOOR TO SWAROOPGANJ_ROW_LHS.mp4'
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
            results_det = model(frame, conf=0.50, iou=0.06 ,imgsz=640, device=0, save=False)
            annotated_frame = results_det[0].plot()

            for result_det in results_det:
                for box in result_det.boxes:
                    # xd,yd,wd,hd = box.xywh.cpu().numpy()[0]
                    # xd,yd,xd,yd = int(xd),int(yd),int(xd),int(yd)
                    # xd1,yd1,xd2,yd2 = (box.xyxy.cpu().numpy()[0])
                    # xd1,yd1,xd2,yd2 = int(xd1),int(yd1),int(xd2),int(yd2)
                    # box_center_label =f'Road Funtiture Centre cordinates are  {str(xd)}  {str(yd)}'
                    class_id_det = box.cls.cpu().numpy()
                    confidence_det = float(box.conf.cpu())
                    class_name_det = result_det.names[int(box.cls)]
                    label_centre_coord = f"{class_name_det}: {confidence_det:.2f}"
                    cv2.putText(annotated_frame, label_centre_coord, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (120, 0, 255), 1)
                    cv2.imshow("Road Furniture Detection", annotated_frame)
                    
                    # if class_id_det is not None:
                    #     print('inside result' , class_id_det)
                    #     time.sleep(0.2)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # output_video.release()
    cap.release()
    cv2.destroyAllWindows()

