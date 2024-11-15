from ultralytics import YOLO
import cv2
import time
import numpy as np
import csv
import pandas as pd
from gps_conversion import GPS_Cord
import warnings
warnings.filterwarnings("ignore")
model_cls = YOLO(r'cls/asp_conc5/weights/best.pt')
import pdb
import pandas as pd
video_path = "K:/survey_data/2d46fac9-fd1d-47e5-ad8d-c087a671c2d2/Akkalkot_SURFACE.mp4"


if __name__ == '__main__':
    total_processing_time = 0
    frame_processing_timeList = []
    frame_processing_timeList = []
    featureType = []
    gps_cord = []
    pixel_cord = []
    fileName = []
    latits = []
    longits = []
    altitudes = []
    ptX = []
    ptY = []
    classIDs = []
    fileName = []
    confList = []
    xd1list = []    
    xd2list = []
    xd3list = []
    xd4list = []
    yd1list = []
    yd2list = []
    yd3list = []
    yd4list = []
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total_frames' , total_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    # output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
    cv2.namedWindow("Road Furniture Detection", cv2.WINDOW_NORMAL)
    frameN = 0
    dataFrame = pd.DataFrame(columns=['Section_ID','RegionIndex', 'data_type','annotation_type','Feature Type' ,'PointsX1','PointsY1','PointsX2','PointsY2','PointsX3','PointsY3','PointsX4','PointsY4','Latitude','Longitude' ,'Altitude','confidence'])  
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            curr_frame =0
            frame1 = frame.copy()
            # frame1[0:600, :] = [0, 0, 0]  # RGB values for black
            startTime = time.time()
            startTimeSegment = time.time()
            results_cls = model_cls(frame1, conf=0.6, iou=0.5, imgsz=640, device=0,show_boxes = False)

            annotated_frame = results_cls[0].plot(boxes=False)
            # annotated_frame[0:601,:] =frame[0:601,:]
            for result_cls in results_cls:
                class_id = result_cls.probs.top1
                class_conf = result_cls.probs.top1conf.cpu()
                # print(result_cls)
                # pdb.set_trace()
                xd1list.append(0)
                xd2list.append(0)
                xd3list.append(0)
                xd4list.append(0)
                yd1list.append(0)
                yd2list.append(0)
                yd3list.append(0)
                yd4list.append(0)
                latits.append(0)
                longits.append(0)
                altitudes.append(0)
                ptX.append(0)
                ptY.append(0)

                fileName.append(0)
                confList.append(0)
                if result_cls.probs.top1 == 1:
                    cv2.putText(frame1, str('Concrete') + str(class_conf), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                    classIDs.append(120)
                    featureType.append('Concrete')
                if result_cls.probs.top1 == 0:
                    cv2.putText(frame1, str('Asphalt') + str (class_conf), (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                    classIDs.append(121)
                    featureType.append('Asphalt')
            cv2.imshow("Road Furniture ROW", frame1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # output_video.release()
    cap.release()
    cv2.destroyAllWindows()

    dataFrame['RegionIndex'] = fileName
    dataFrame['annotation_type'] = classIDs
    dataFrame['Feature Type'] = featureType

    dataFrame['Latitude'] = latits
    dataFrame['Longitude'] = longits  
    dataFrame['Altitude'] = altitudes
    dataFrame['confidence'] = confList
    dataFrame['Section_ID'] = "8b38d31c-becb-445c-bb11-6eb5f61ed24d"
    dataFrame['Section_ID'].replace('','8b38d31c-becb-445c-bb11-6eb5f61ed24d', inplace=True)
    dataFrame['data_type'] = '00'
    dataFrame['data_type'].replace('','00', inplace=True)
    dataFrame['PointsX1'] = pd.Series(xd1list)
    dataFrame['PointsY1'] = pd.Series(yd1list)
    dataFrame['PointsX2'] = pd.Series(xd2list)
    dataFrame['PointsY2'] = pd.Series(yd2list)
    dataFrame['PointsX3'] = pd.Series(xd3list)
    dataFrame['PointsY3'] = pd.Series(yd3list)
    dataFrame['PointsX4'] = pd.Series(xd4list)
    dataFrame['PointsY4'] = pd.Series(yd4list)
    dataFrame.to_csv('outer_demo_row_trial_8b38d31c-becb-445c-bb11-6eb5f61ed24d.csv')