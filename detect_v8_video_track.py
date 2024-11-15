from ultralytics import YOLO
import cv2
import time
import numpy as np
import csv
import pandas as pd
from gps_conversion import GPS_Cord

import warnings
warnings.filterwarnings("ignore")

fn1 = 'K:/3b218544-91c0-425d-aa17-ea0e0d716d2f/d5ef3a58-01e2-4548-82b8-c0d3ff381763/interpolated_track.pkl'
I = pd.read_pickle(fn1)
fn2 = 'K:/3b218544-91c0-425d-aa17-ea0e0d716d2f/d5ef3a58-01e2-4548-82b8-c0d3ff381763/pcams/LL.log'
E = pd.read_csv(fn2,sep="/",names = ['c1'], usecols=[7])
model_det = YOLO(r'roadFurniture/7mar/weights/best.pt')
model_seg = YOLO(r'segmentation/7mar/weights/best.pt')
video_path = "K:/3b218544-91c0-425d-aa17-ea0e0d716d2f/out_YDSHI-AURANGABAD_ROW_LHS_INNER.mp4"


if __name__ == '__main__':
    total_processing_time = 0
    frame_processing_timeList = []
    featureType = []
    gps_cord = []
    pixel_cord = []
    fileName = []

    cap = cv2.VideoCapture(video_path)
    # output_video_name = "outer_video_demo_d658e3cc-f3a3-42f5-aca0-5c37de805475.mp4"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total_frames' , total_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    # output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
    cv2.namedWindow("Road Furniture outer", cv2.WINDOW_NORMAL)
    frameN = 0
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            mask_frame = frame.copy()
            height, width = frame.shape[:2]
            start_col = 0
            # end_col = width - (width // 3)
            # mask_frame[:,end_col:width] = [0, 0, 0]  # RGB values for black
            # cv2.imwrite('frame.jpg', mask_frame)

            curr_frame =0
            startTime = time.time()
            startTimeSegment = time.time()
            results_segm = model_seg.track(mask_frame, conf=0.4, iou=0.47, imgsz=640, device=0,show_boxes = False, persist=True)
            annotated_frame = results_segm[0].plot()
            # annotated_frame[:,end_col:width] =frame[:,end_col:width]  # RGB values for black
            for result_seg in results_segm:
                if result_seg.masks is not None:
                    # annotated_frame[:,end_col:width] =frame[:,end_col:width]  # RGB values for black
                    for mask, box in zip(result_seg.masks, result_seg.boxes):
                        class_id = int(box.cls[0])  # Access to the class ID
                        confidence = box.conf[0]    # Access to the confidence
                        class_name = result_seg.names[class_id]  # Class name based on the ID
                        label = f"{class_name}: {confidence:.2f}"
                        box_coords_xyxy = box.xyxy.cpu().numpy()         ###CORDINATES BOX
                        box_coords_xywh = box.xywh.cpu().numpy()
                        xs1, ys1 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1])  
                        xsc, ysc = int(box_coords_xywh[0][0]), int(box_coords_xywh[0][1])
                        # obj = GPS_Cord([(xsc,ysc)],I[['lat','long','hell','pitch','roll','head']].iloc[frameN])
                        # gps_loc = obj.geodetic2ecef()  
                        # import pdb
                        # pdb.set_trace()
                        box_center_label =f'Centre pixel cordinates are  {str(xsc)}  {str(ysc)}'
                        points = mask.xy.copy()
                        points = [np.asarray(point) for point in points]
                        points = [np.round(point).astype(np.int32) for point in points]

                        # points = np.round(points).astype(np.int32)
                        # points_for_cv2 = points[:, np.newaxis, :]                        
                        for contour in points:
                            convexHull = cv2.convexHull(contour)
                            cv2.drawContours(annotated_frame, [convexHull], -1, (255, 255, 255), 3)   

                        rect = cv2.minAreaRect(np.concatenate(points, axis=0))
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        center_point = np.mean(box, axis=0, dtype=np.int0)
                        # annotated_frame = cv2.drawContours(annotated_frame.copy(), [box], 0, (0, 255, 0), 4)
                        x_boxC , y_boxC = center_point

                        box_center_label =f'{class_name} Centre pixel cordinates are  {str(x_boxC)}  {str(y_boxC)}'
                        obj = GPS_Cord([(x_boxC,y_boxC)],I[['lat','long','hell','pitch','roll','head']].iloc[frameN])
                        gps_loc = obj.geodetic2ecef()  
                        # cv2.circle(annotated_frame, tuple(center_point), 5, (0, 0, 255), -1)


                        featureType.append(class_name)
                        gps_cord.append((gps_loc['latitude'][0],gps_loc['longitude'][0]))
                        pixel_cord.append((xsc, ysc))
                        fileName.append(E['c1'].iloc[frameN])

                        road_furniture_lat_long = f"{class_name} Barrier location {gps_loc}"
                        # cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                        '''if class_id == 0:
                            cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                            cv2.putText(annotated_frame, label, (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                            cv2.putText(annotated_frame,box_center_label, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                        elif class_id == 1:   
                            cv2.putText(annotated_frame, label, (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                            cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                            cv2.putText(annotated_frame, box_center_label, (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)  
                        elif class_id == 2:   
                            cv2.putText(annotated_frame, label, (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                            cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                            cv2.putText(annotated_frame, box_center_label, (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)    '''  
            endTimeSegment = time.time()
            segmentTime = endTimeSegment-startTimeSegment
            print("segmentTime processing time: {:.4f} seconds".format(segmentTime))

            startTimeDetect = time.time()
            results_det = model_det.track(mask_frame, conf=0.40, imgsz=1024, persist=True,device=0)
            for result_det in results_det:
                for box in result_det.boxes:
                    xd,yd,wd,hd = box.xywh.cpu().numpy()[0]
                    xd,yd,xd,yd = int(xd),int(yd),int(xd),int(yd)
                    xd1,yd1,xd2,yd2 = (box.xyxy.cpu().numpy()[0])
                    xd1,yd1,xd2,yd2 = int(xd1),int(yd1),int(xd2),int(yd2)
                    if xd1 > 1500:
                        continue
                    box_center_label =f'Road Funtiture Centre cordinates are  {str(xd)}  {str(yd)}'
                    class_id_det = box.cls.cpu().numpy()
                    track_id = box.id.cpu().numpy()
                    class_name_det = result_det.names[int(box.cls)]
                    confidence_det = float(box.conf.cpu())
                    label_det = f"{track_id} -{class_name_det}: {confidence_det:.2f}"
                    label_centre_coord = f"{class_name_det}: {confidence_det:.2f}"
                    cv2.putText(annotated_frame, label_det, (xd1, yd1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
                    cv2.putText(annotated_frame, box_center_label, (1500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

                    cv2.rectangle(annotated_frame, (xd1,yd1), (xd2,yd2), (120, 0, 255), 4) 

                    obj = GPS_Cord([(xd,yd)],I[['lat','long','hell','pitch','roll','head']].iloc[frameN])
                    gps_loc = obj.geodetic2ecef()
                    road_furniture_lat_long = f"Road Furniture location {gps_loc}"
                    cv2.putText(annotated_frame, str(road_furniture_lat_long), (1200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
                    featureType.append(class_name_det)
                    gps_cord.append((gps_loc['latitude'][0],gps_loc['longitude'][0]))
                    pixel_cord.append((xd, yd))
                    fileName.append(E['c1'].iloc[frameN])
            cv2.putText(annotated_frame, str(E['c1'].iloc[frameN]), (1300, 2000), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
            endTimeDetect = time.time()
            detectTime = endTimeDetect-startTimeDetect
            print("DETECT processing time: {:.4f} seconds".format(detectTime))

            endTime = time.time()
            # output_video.write(annotated_frame)
            
            frameProcessTime = endTime - startTime
            print("Frame processing time: {:.4f} seconds".format(frameProcessTime))
            total_processing_time += frameProcessTime
            frame_processing_timeList.append(frameProcessTime)
            cv2.imshow("Road Furniture outer", annotated_frame)
            frameN = frameN + 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # output_video.release()
    cap.release()
    cv2.destroyAllWindows()

    # Calculate and print the average processing time per frame
    average_frame_processing_time = total_processing_time / total_frames
    print("Average processing time per frame: {:.4f} seconds".format(average_frame_processing_time))
    dataFrame = pd.DataFrame(columns=['Frame Number','Process Time', 'File Name', 'Feature Type' ,'Pixel Cordinates', 'Gps cordinates'])

    # dataFrame['Frame Number'] = frameN
    # dataFrame['Process Time'] = frame_processing_timeList
    dataFrame['File Name'] = fileName
    dataFrame['Feature Type'] = featureType
    dataFrame['Pixel Cordinates'] = pixel_cord
    dataFrame['Gps cordinates'] = gps_cord  

    # dataFrame.to_csv('outer_demo_d658e3cc-f3a3-42f5-aca0-5c37de805475.csv')
    # # Write frame processing times to a CSV file
    # with open('outer_demo1.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Frame Number', 'Processing Time (seconds)'])
    #     for i, time in enumerate(frame_processing_timeList):
    #         writer.writerow([i+1, time])