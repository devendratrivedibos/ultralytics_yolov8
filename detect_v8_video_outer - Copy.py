from ultralytics import YOLO
import cv2
import time
import numpy as np
import csv
import pandas as pd
from gps_conversion import GPS_Cord


import warnings
warnings.filterwarnings("ignore")


fn1 = 'F:/81ff0c01-4a98-441d-8c71-a782c29bf9a2/25b8d909-75d6-44f5-add1-cba58aa1c198/interpolated_track.pkl'
I = pd.read_pickle(fn1)
fn2 = 'F:/81ff0c01-4a98-441d-8c71-a782c29bf9a2/25b8d909-75d6-44f5-add1-cba58aa1c198/pcams/LL.log'
E = pd.read_csv(fn2,sep="/",names = ['c1'], usecols=[7])
# import pdb
# pdb.set_trace()
model_det = YOLO(r'roadFurniture/16mar/weights/best.pt')

# model_seg = YOLO(r'segmentation/5mar2/weights/best.pt')


model_seg = YOLO(r'segmentation/7mar/weights/best.pt')
# model_seg = YOLO(r'segmentation/15mar/weights/best.pt')
video_path = "F:/81ff0c01-4a98-441d-8c71-a782c29bf9a2/KHED-SINNAR_ROW_10.mp4"
if __name__ == '__main__':
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
    cap = cv2.VideoCapture(video_path)
    # output_video_name = "trial_25b8d909-75d6-44f5-add1-cba58aa1c198.mp4"
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
            end_col = width - (width // 3)
            mask_frame[:,end_col:width] = [0, 0, 0]  # RGB values for black
            # cv2.imwrite('frame.jpg', mask_frame)

            curr_frame =0
            startTime = time.time()
            startTimeSegment = time.time()
            results_segm = model_seg(mask_frame, conf=0.5, iou=0.47, imgsz=640, device=0,show_boxes = False, classes = [0,1])
            annotated_frame = results_segm[0].plot(boxes = False)
            
            # cv2.putText(annotated_frame, str(E['c1'].iloc[frameN]), (500, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            annotated_frame[:,end_col:width] =frame[:,end_col:width]  # RGB values for black
            for result_seg in results_segm:
                if result_seg.masks is not None:
                    annotated_frame[:,end_col:width] =frame[:,end_col:width]  # RGB values for black
                    for mask, box in zip(result_seg.masks, result_seg.boxes):
                        class_id = int(box.cls[0])  # Access to the class ID
                        confidence = float(box.conf[0].cpu())    # Access to the confidence
                        class_name = result_seg.names[class_id]  # Class name based on the ID
                        label = f"{class_name}: {confidence:.2f}"
                        box_coords_xyxy = box.xyxy.cpu().numpy()         ###CORDINATES BOX
                        box_coords_xywh = box.xywh.cpu().numpy()
                        xs1, ys1 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1])  
                        xsc, ysc = int(box_coords_xywh[0][0]), int(box_coords_xywh[0][1])
                        # xd1list.append(0)
                        # xd2list.append(0)
                        # xd3list.append(0)
                        # xd4list.append(0)
                        # yd1list.append(0)
                        # yd2list.append(0)
                        # yd3list.append(0)
                        # yd4list.append(0)

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
                        # print(box[0] , box[0][0],box[0][1])
                        # print(box[1],box[1][0],box[1][1])
                        # print(box[2],box[2][0],box[2][1])
                        # print(box[3],box[3][0],box[3][1])
                        xd1 , yd1 = (box[0][0],box[0][1])
                        xd2 , yd2= (box[1][0],box[1][1])
                        xd3 , yd3= (box[2][0],box[2][1])
                        xd4 , yd4 = (box[3][0],box[3][1])
                        xd1list.append(xd1)
                        xd2list.append(xd2)
                        xd3list.append(xd3)
                        xd4list.append(xd4)
                        yd1list.append(yd1)
                        yd2list.append(yd2)
                        yd3list.append(yd3)
                        yd4list.append(yd4)
                        center_point = np.mean(box, axis=0, dtype=np.int0)
                        # annotated_frame = cv2.drawContours(annotated_frame.copy(), [box], 0, (0, 255, 0), 4)
                        x_boxC , y_boxC = center_point

                        box_center_label =f'{class_name} Centre pixel cordinates are  {str(x_boxC)}  {str(y_boxC)}'
                        # obj = GPS_Cord([(x_boxC,y_boxC)],I[['lat','long','hell','pitch','roll','head']].iloc[frameN])
                        # gps_loc = obj.geodetic2ecef()  
                        # cv2.circle(annotated_frame, tuple(center_point), 5, (0, 0, 255), -1)


                        featureType.append(class_name)
                        pixel_cord.append((xsc, ysc))
                        # fileName.append(E['c1'].iloc[frameN])
                        # fileName.append(E['c1'].iloc[frameN].split('-')[1])
                        fileName.append(frameN)
                        confList.append(confidence)
                        # gps_cord.append((gps_loc['latitude'][0],gps_loc['longitude'][0],gps_loc['altitude'][0]))
                        # latits.append(gps_loc['latitude'][0])
                        # longits.append(gps_loc['longitude'][0])
                        # altitudes.append(gps_loc['altitude'][0])
                        gps_loc = I[['Latitude','Longitude','Altitude']].iloc[frameN]
                        # pdb.set_trace()
                        # gps_cord.append((gps_loc['Latitude'][0],gps_loc['Longitude'][0],gps_loc['Altitude'][0]))
                        # latits.append(gps_loc['Latitude'][0])
                        # longits.append(gps_loc['Longitude'][0])
                        # altitudes.append(gps_loc['Altitude'][0])
                        latits.append(gps_loc['Latitude'])
                        longits.append(gps_loc['Longitude'])
                        altitudes.append(gps_loc['Altitude'])
                        ptX.append(xsc)
                        ptY.append(ysc)
                        classIDs.append(class_id)
                        road_furniture_lat_long = f"{class_name} Barrier location {gps_loc}"
                        # cv2.putText(annotated_frame, str(road_furniture_lat_long), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
                        if class_id == 0:
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
                            cv2.putText(annotated_frame, box_center_label, (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)      
            endTimeSegment = time.time()
            segmentTime = endTimeSegment-startTimeSegment
            print("segmentTime processing time: {:.4f} seconds".format(segmentTime))

            startTimeDetect = time.time()
            results_det = model_det(mask_frame, conf=0.40, imgsz=1024, device=0)
            for result_det in results_det:
                for box in result_det.boxes:
                    xd,yd,wd,hd = box.xywh.cpu().numpy()[0]
                    xd,yd,xd,yd = int(xd),int(yd),int(xd),int(yd)

                    xd1,yd1,xd4,yd4 = (box.xyxy.cpu().numpy()[0])
                    xd1,yd1,xd4,yd4 = int(xd1),int(yd1),int(xd4),int(yd4)
                    xd3 = xd1
                    xd2 = xd4
                    yd3 = yd4
                    yd2 = yd1


                    # if xd1 > 1500:
                    #     continue
                    box_center_label =f'Road Funtiture Centre cordinates are  {str(xd)}  {str(yd)}'
                    class_id_det = int(box.cls.cpu().numpy())
                    class_name_det = result_det.names[int(box.cls)]
                    confidence_det = float(box.conf.cpu())
                    label_det = f"{class_name_det}: {confidence_det:.2f}"
                    label_centre_coord = f"{class_name_det}: {confidence_det:.2f}"
                    cv2.putText(annotated_frame, label_det, (xd1, yd1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
                    cv2.putText(annotated_frame, box_center_label, (1500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

                    cv2.rectangle(annotated_frame, (xd1,yd1), (xd2,yd2), (120, 0, 255), 4) 

                    # obj = GPS_Cord([(xd,yd)],I[['lat','long','hell','pitch','roll','head']].iloc[frameN])
                    # gps_loc = obj.geodetic2ecef()
                    # road_furniture_lat_long = f"Road Furniture location {gps_loc}"
                    # cv2.putText(annotated_frame, str(road_furniture_lat_long), (1200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
                    featureType.append(class_name_det)
                    confList.append(confidence_det)

                    gps_loc = I[['Latitude','Longitude','Altitude']].iloc[frameN]
                    # pdb.set_trace()
                    # gps_cord.append((gps_loc['Latitude'][0],gps_loc['Longitude'][0],gps_loc['Altitude'][0]))
                    # latits.append(gps_loc['Latitude'][0])
                    # longits.append(gps_loc['Longitude'][0])
                    # altitudes.append(gps_loc['Altitude'][0])
                    latits.append(gps_loc['Latitude'])
                    longits.append(gps_loc['Longitude'])
                    altitudes.append(gps_loc['Altitude'])
                    pixel_cord.append((xd, yd))
                    fileName.append(frameN)
                    # fileName.append(E['c1'].iloc[frameN].split('-')[1])
                    # gps_cord.append((gps_loc['Latitude'][0],gps_loc['Longitude'][0],gps_loc['Altitude'][0]))
                    # latits.append(gps_loc['Latitude'][0])
                    # longits.append(gps_loc['Longitude'][0])
                    # altitudes.append(gps_loc['Altitude'][0])
                    # ptX.append(xd)
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
            # print('Image is ',str(E['c1'].iloc[frameN]))
            # cv2.putText(annotated_frame, str(E['c1'].iloc[frameN]), (1300, 2000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
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
    print(f"total_processing_time-------- {total_processing_time}" )
    print("Average processing time per frame: {:.4f} seconds".format(average_frame_processing_time))
    # dataFrame = pd.DataFrame(columns=['Frame Number','Process Time', 'File Name', 'Feature Type' ,'Confidence Level','Pixel Cordinates', 'Gps cordinates'])

    # dataFrame = pd.DataFrame(columns=['RegionIndex', 'annotation_type','Feature Type' ,'PointsX','PointsY','Latitude','Longitude' , 'confidence'])
    dataFrame = pd.DataFrame(columns=['Section_ID','RegionIndex', 'data_type','annotation_type','Feature Type' ,'PointsX1','PointsY1','PointsX2','PointsY2','PointsX3','PointsY3','PointsX4','PointsY4','Latitude','Longitude' ,'Altitude','confidence'])  

    # dataFrame['Frame Number'] = frameN
    # dataFrame['Process Time'] = frame_processing_timeList
    dataFrame['RegionIndex'] = pd.Series(fileName)
    dataFrame['annotation_type'] = pd.Series(classIDs)
    dataFrame['Feature Type'] = pd.Series(featureType)

    dataFrame['Latitude'] = latits
    dataFrame['Longitude'] = longits  
    dataFrame['Altitude'] = altitudes
    dataFrame['confidence'] = pd.Series(confList)
    dataFrame['Section_ID'] = "25b8d909-75d6-44f5-add1-cba58aa1c198"
    dataFrame['Section_ID'].replace('','25b8d909-75d6-44f5-add1-cba58aa1c198', inplace=True)
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

    dataFrame.to_csv('25b8d909-75d6-44f5-add1-cba58aa1c198.csv')
    # # Write frame processing times to a CSV file
    # with open('outer_demo1.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Frame Number', 'Processing Time (seconds)'])
    #     for i, time in enumerate(frame_processing_timeList):
    #         writer.writerow([i+1, time])