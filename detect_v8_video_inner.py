from ultralytics import YOLO
import cv2
import time
import numpy as np
import csv
import pandas as pd
from gps_conversion import GPS_Cord
import pdb
import warnings
warnings.filterwarnings("ignore")
import pdb



# 'f62d5074-9cdc-451e-8720-466fa51a13ab'

projectI = 'D:/LUCKNOW-AYODHYA_2024-08-30_09-10-40'
sectionID = '8caf99f8-4d81-408c-809a-79453a821daf'
# sectionID = '5dd6fcd1-4142-4f90-800b-8429dbdf9da6'
fn1 = f'{projectI}/{sectionID}/interpolated_track.pkl'
I = pd.read_pickle(fn1)
# fn2 = f'{projectI}/{sectionID}/pcams/LL.log'
# E = pd.read_csv(fn2,sep="/",names = ['c1'], usecols=[7])


model_det = YOLO(r'roadFurniture/16mar/weights/best.pt')
model_seg = YOLO(r'segmentation/7mar/weights/best.pt')
# model_seg = YOLO(r'segmentation/15mar/weights/best.pt')
video_path = f"{projectI}/ROW/KHED-SINNAR_ROW_0.mp4"

if __name__ == '__main__':
    # sectionID = ['5aa54a71-4d72-4c49-bd1a-89665dd1f8b1']
    total_processing_time = 0
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
    # pdb.set_trace()
    cap = cv2.VideoCapture(video_path)
    # output_video_name = "Inner_demo_trial_345f29d5-b9aa-4c79-9fa6-fdcffbd57cd8.mp4"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total_frames' , total_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    # output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
    cv2.namedWindow("Road Furniture Detection", cv2.WINDOW_NORMAL)
    frameN = 0
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            mask_frame = frame.copy()
            
            height, width = frame.shape[:2]
            start_col = 0
            end_col = width // 3
            mask_frame[:, start_col:end_col] = [0, 0, 0]  # RGB values for black
            # mask_frame[:,end_col:width] = [0, 0, 0]  # RGB values for black     
            
            curr_frame =0
            startTime = time.time()
            startTimeSegment = time.time()
            results_segm = model_seg(mask_frame, conf=0.4, iou=0.47, imgsz=640, device=0,show_boxes = False)
            annotated_frame = results_segm[0].plot(boxes = False)
            annotated_frame[:, start_col:end_col] =frame[:, start_col:end_col]  # RGB         values for black
            print(annotated_frame.shape)
            # annotated_frame[:,end_col:width] =frame[:,end_col:width]
            for result_seg in results_segm:
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
                        # pdb.set_trace()
                        # obj = GPS_Cord([(xsc,ysc)],I[['lat','long','hell','pitch','roll','head']].iloc[frameN])
                        # # obj = GPS_Cord([(xd,yd)],I[['Latitude','Longitude','Altitude']].iloc[frameN])
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
                        xd1 , yd1 = (box[0][0],box[0][1])
                        xd2 , yd2= (box[1][0],box[1][1])
                        xd3 , yd3= (box[2][0],box[2][1])
                        xd4 , yd4 = (box[3][0],box[3][1])
                        center_point = np.mean(box, axis=0, dtype=np.int0)
                        # annotated_frame = cv2.drawContours(annotated_frame.copy(), [box], 0, (0, 255, 0), 4)
                        x_boxC , y_boxC = center_point

                        box_center_label =f'{class_name} Centre pixel cordinates are  {str(x_boxC)}  {str(y_boxC)}'

                        ###
                        # obj = GPS_Cord([(x_boxC,y_boxC)],I[['lat','long','hell','pitch','roll','head']].iloc[frameN])
                        # gps_loc = obj.geodetic2ecef()  
                        ###

                        # cv2.circle(annotated_frame, tuple(center_point), 5, (0, 0, 255), -1)
                        xd1list.append(xd1)
                        xd2list.append(xd2)
                        xd3list.append(xd3)
                        xd4list.append(xd4)
                        yd1list.append(yd1)
                        yd2list.append(yd2)
                        yd3list.append(yd3)
                        yd4list.append(yd4)

                        featureType.append(class_name)
                        pixel_cord.append((xsc, ysc))
                        # fileName.append(E['c1'].iloc[frameN].split('-')[1])
                        fileName.append(frameN)
                        confList.append(confidence)
                        # gps_loc = I[['Latitude','Longitude','Altitude']].iloc[frameN]
                        gps_loc = I[['lat','long','hell']].iloc[frameN]
                        latits.append(gps_loc['lat'])
                        longits.append(gps_loc['long'])
                        altitudes.append(gps_loc['hell'])
                        # pdb.set_trace()
                        # gps_cord.append((gps_loc['Latitude'][0],gps_loc['Longitude'][0],gps_loc['Altitude'][0]))
                        # latits.append(gps_loc['Latitude'][0])
                        # longits.append(gps_loc['Longitude'][0])
                        # altitudes.append(gps_loc['Altitude'][0])
                        # latits.append(gps_loc['Latitude'])
                        # longits.append(gps_loc['Longitude'])
                        # altitudes.append(gps_loc['Altitude'])

                        ptX.append(xsc)
                        ptY.append(ysc)
                        classIDs.append(class_id)
                        road_furniture_lat_long = f"{class_name} Barrier location {gps_loc}"
                        # cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                        if class_id == 0:
                            cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                            # cv2.putText(annotated_frame, label, (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                            cv2.putText(annotated_frame,box_center_label, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                        elif class_id == 1:   
                            cv2.putText(annotated_frame, label, (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                            # cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                            cv2.putText(annotated_frame, box_center_label, (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)  
                        elif class_id == 2:   
                            cv2.putText(annotated_frame, label, (xs1, ys1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                            # cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
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


                    box_center_label =f'Road Funtiture Centre cordinates are  {str(xd)}  {str(yd)}'
                    class_id_det = int(box.cls.cpu().numpy())
                    class_name_det = result_det.names[int(box.cls)]
                    confidence_det = float(box.conf.cpu())
                    label_det = f"{class_name_det}: {confidence_det:.2f}"
                    label_centre_coord = f"{class_name_det}: {confidence_det:.2f}"
                    # if class_id_det == 11 and confidence_det < 0.8:              ####go slow
                    #     continue
                    # if class_id_det == 34 and confidence_det < 0.8:              ###accidnt zone
                    #     continue
                    # if class_id_det == 19 and confidence_det < 0.8:              ###stop
                    #     continue

                    cv2.putText(annotated_frame, label_det, (xd1, yd1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
                    cv2.putText(annotated_frame, box_center_label, (1500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                    cv2.rectangle(annotated_frame, (xd1,yd1), (xd2,yd2), (120, 0, 255), 4) 
                    # pdb.set_trace()
                    # obj = GPS_Cord([(xd,yd)],I[['Latitude','Longitude','Altitude','pitch','roll','head']].iloc[frameN])
                    # obj = GPS_Cord([(xd,yd)],I[['Latitude','Longitude','Altitude']].iloc[frameN])
                    # gps_loc = obj.geodetic2ecef()
                    # gps_loc = I[['Latitude','Longitude','Altitude']].iloc[frameN]
                    gps_loc = I[['lat','long','hell']].iloc[frameN]
                    latits.append(gps_loc['lat'])
                    longits.append(gps_loc['long'])
                    altitudes.append(gps_loc['hell'])
                    # latits.append(gps_loc['Latitude'])
                    # longits.append(gps_loc['Longitude'])
                    # altitudes.append(gps_loc['Altitude'])
                    road_furniture_lat_long = f"Road Furniture location {gps_loc}"
                    
                    cv2.putText(annotated_frame, str(road_furniture_lat_long), (1200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
                    featureType.append(class_name_det)

                    pixel_cord.append((xd, yd))
                    # fileName.append(E['c1'].iloc[frameN])
                    # pdb.set_trace()
                    fileName.append(frameN)
                    # fileName.append(E['c1'].iloc[frameN].split('-')[1])
                    confList.append(confidence_det)

                    
                    # pdb.set_trace()
                    # gps_cord.append((gps_loc['Latitude'][0],gps_loc['Longitude'][0],gps_loc['Altitude'][0]))

                    ptX.append(xd)
                    ptY.append(yd)
                    xd1list.append(xd1)
                    xd2list.append(xd2)
                    xd3list.append(xd3)
                    xd4list.append(xd4)
                    yd1list.append(yd1)
                    yd2list.append(yd2)
                    yd3list.append(yd3)
                    yd4list.append(yd4)
                    classIDs.append(class_id_det)
            # cv2.putText(annotated_frame, str(E['c1'].iloc[frameN]), (1300, 2000), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
            endTimeDetect = time.time()
            detectTime = endTimeDetect-startTimeDetect
            print("DETECT processing time: {:.4f} seconds".format(detectTime))
            results_segm = None
            results_det = None
            endTime = time.time()
            # output_video.write(annotated_frame)
            
            frameProcessTime = endTime - startTime
            print("Frame processing time: {:.4f} seconds".format(frameProcessTime))
            total_processing_time += frameProcessTime
            frame_processing_timeList.append(frameProcessTime)
            cv2.imshow("Road Furniture Detection", annotated_frame)
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
    dataFrame = pd.DataFrame(columns=['Section_ID','RegionIndex', 'data_type','annotation_type','Feature Type' ,'PointsX1','PointsY1','PointsX2','PointsY2','PointsX3','PointsY3','PointsX4','PointsY4','Latitude','Longitude' ,'Altitude','confidence'])  

    # dataFrame['Frame Number'] = frameN
    # dataFrame['Process Time'] = frame_processing_timeList
    # print(fileName)
    # dataFrame.loc[:, 'Section_id'] = r"0341ae19-ccf7-4cf5-9a52-14f4f9319e0a"
    dataFrame['RegionIndex'] = pd.Series(fileName)
    dataFrame['annotation_type'] = pd.Series(classIDs)
    dataFrame['Feature Type'] = pd.Series(featureType)
    
    dataFrame['PointsX1'] = pd.Series(xd1list)
    dataFrame['PointsY1'] = pd.Series(yd1list)
    dataFrame['PointsX2'] = pd.Series(xd2list)
    dataFrame['PointsY2'] = pd.Series(yd2list)
    dataFrame['PointsX3'] = pd.Series(xd3list)
    dataFrame['PointsY3'] = pd.Series(yd3list)
    dataFrame['PointsX4'] = pd.Series(xd4list)
    dataFrame['PointsY4'] = pd.Series(yd4list)

    dataFrame['Latitude'] = pd.Series(latits)
    dataFrame['Longitude'] = pd.Series(longits) 
    dataFrame['Altitude'] = pd.Series(altitudes)
    dataFrame['confidence'] =  pd.Series(confList)
    dataFrame['data_type'] = '00'
    dataFrame['data_type'].replace('','00', inplace=True)

    dataFrame['Section_ID'] = sectionID
    dataFrame['Section_ID'].replace('',sectionID, inplace=True)
    dataFrame.to_csv(f'{projectI}/{sectionID}/{sectionID}.csv')
    # # Write frame processing times to a CSV file
    # with open('frame_processing_times_demo.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Frame Number', 'Processing Time (seconds)'])
    #     for i, time in enumerate(frame_processing_timeList):
    #         writer.writerow([i+1, time])


    