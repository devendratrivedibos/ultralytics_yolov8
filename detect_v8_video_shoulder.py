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

# fn1 = 'F:/11e81bba-b4a7-4f9e-997a-52783e6a6c3d/1610c09b-6ecd-4aaa-815d-7c9092f22a61/interpolated_track.pkl'
# I = pd.read_pickle(fn1)
# fn2 = 'F:/11e81bba-b4a7-4f9e-997a-52783e6a6c3d/1610c09b-6ecd-4aaa-815d-7c9092f22a61/pcams/LL.log'
# E = pd.read_csv(fn2,sep="/",names = ['c1'], usecols=[7])


# # model_seg = YOLO(r'segmentation/5mar2/weights/best.pt')
# model_seg = YOLO(r'shoulderWidth/21mar/weights/best.pt')

# video_path = "F:/11e81bba-b4a7-4f9e-997a-52783e6a6c3d/SHEDUNG TO ARIWALI_ROW_6.mp4"
fn1 = 'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/0f2bd68e-ddd9-4345-905f-563a573cb181/'
fn2 = 'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/0f2bd68e-ddd9-4345-905f-563a573cb181/pcams/LL.log'
E = pd.read_csv(fn2,sep="/",names = ['c1'], usecols=[7])


# model_seg = YOLO(r'segmentation/5mar2/weights/best.pt')
model_seg = YOLO(r'shoulderWidth/21mar/weights/best.pt')

video_path = r"E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/OUT_Hydrabad_ROW_Outer.mp4"



projectI = 'K:/046376c9-a9ae-4126-b38e-07a55090024c'
sectionID = 'fb550257-e695-4120-bf50-85ec6c40afb4'
# sectionID = '5dd6fcd1-4142-4f90-800b-8429dbdf9da6'
fn1 = f'{projectI}/{sectionID}/interpolated_track.pkl'
I = pd.read_pickle(fn1)
# fn2 = f'{projectI}/{sectionID}/pcams/LL.log'
# E = pd.read_csv(fn2,sep="/",names = ['c1'], usecols=[7])


model_det = YOLO(r'roadFurniture/16mar/weights/best.pt')
model_seg = YOLO(r'segmentation/7mar/weights/best.pt')


model_seg = YOLO(r'shoulderWidth/21mar/weights/best.pt')
fn1 = 'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/0f2bd68e-ddd9-4345-905f-563a573cb181/'
fn2 = 'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/0f2bd68e-ddd9-4345-905f-563a573cb181/pcams/LL.log'
E = pd.read_csv(fn2,sep="/",names = ['c1'], usecols=[7])


# model_seg = YOLO(r'segmentation/5mar2/weights/best.pt')
model_seg = YOLO(r'shoulderWidth/21mar/weights/best.pt')

video_path = r"E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/OUT_Hydrabad_ROW_Outer.mp4"


if __name__ == '__main__':
    total_processing_time = 0
    frame_processing_timeList = []
    featureType = []
    gps_cord = []
    latits = []
    longits = []
    pixel_cord = []
    ptX = []
    ptY = []
    fileName = []


    cap = cv2.VideoCapture(video_path)
    output_video_name = "Inner_demo_5d74a272-d4fb-4cc8-a02b-3ab23e50ecd0.mp4"
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
        try:
            if success:
                mask_frame = frame.copy()
                
                x1_diff_list_road = []
                y1_diff_list_road = []
                x2_diff_list_road = []
                y2_diff_list_road = []

                x1_diff_list_lane = []
                y1_diff_list_lane = []
                x2_diff_list_lane = []
                y2_diff_list_lane = []
                
                curr_frame =0
                startTime = time.time()
                startTimeSegment = time.time()
                results_segm = model_seg(mask_frame, conf=0.5, iou=0.8, imgsz=640, device=0,show_boxes = False)
                annotated_frame = results_segm[0].plot(boxes = False)

                for result_seg in results_segm:
                    if result_seg.masks is not None:
                        # annotated_frame[:, start_col:end_col] =frame[:, start_col:end_col]  # RGB values for black
                        for mask, box in zip(result_seg.masks, result_seg.boxes):

                            class_id = int(box.cls[0])  # Access to the class ID
                            confidence = box.conf[0]    # Access to the confidence
                            class_name = result_seg.names[class_id]  # Class name based on the ID
                            label = f"{class_name}: {confidence:.2f}"
                            box_coords_xyxy = box.xyxy.cpu().numpy()         ###CORDINATES BOX
                            box_coords_xywh = box.xywh.cpu().numpy()
                            xsc, ysc = int(box_coords_xywh[0][0]), int(box_coords_xywh[0][1])
                            xs1, ys1,xs2,ys2 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1]),int(box_coords_xyxy[0][2]),int(box_coords_xyxy[0][3])
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
                            # obj = GPS_Cord([(x_boxC,y_boxC)],I[['lat','long','hell','pitch','roll','head']].iloc[frameN])
                            # gps_loc = obj.geodetic2ecef()  
                            # cv2.circle(annotated_frame, tuple(center_point), 5, (0, 0, 255), -1)


                            featureType.append(class_name)
                            # gps_cord.append((gps_loc['latitude'][0],gps_loc['longitude'][0]))
                            pixel_cord.append((xsc, ysc))
                            # fileName.append(E['c1'].iloc[frameN])
                            # latits.append(gps_loc['latitude'][0])
                            # longits.append(gps_loc['longitude'][0])
                            ptX.append(xsc)
                            ptY.append(ysc)
                            # road_furniture_lat_long = f"{class_name} Barrier location {gps_loc}"
                            

                            if class_id == 1:
                                x1_diff_list_road.append(xs1)
                                y1_diff_list_road.append(ys1)
                                x2_diff_list_road.append(xs2)
                                y2_diff_list_road.append(ys2)
                            if class_id == 0:
                                x1_diff_list_lane.append(xs1)
                                y1_diff_list_lane.append(ys1)
                                x2_diff_list_lane.append(xs2)
                                y2_diff_list_lane.append(ys2)

                endTimeSegment = time.time()
                segmentTime = endTimeSegment-startTimeSegment
                print("segmentTime processing time: {:.4f} seconds".format(segmentTime))


                print(f"x_diff_list_road before sotring" , {str(x1_diff_list_road)})
                print(f"x_diff_list_lane before sotring" , {str(x1_diff_list_lane)})
                x1_diff_list_road.sort()
                x1_diff_list_lane.sort()
                x2_diff_list_road.sort()
                x2_diff_list_lane.sort()
                # # y1_diff_list_road.sort()
                # y1_diff_list_lane.sort()
                # # y2_diff_list_road.sort()
                # y2_diff_list_lane.sort()
                # y_diff_list_road.sort()
                # y_diff_list_lane.sort()
                print(f"x1_diff_list_road Afterrrrr sotring" , {str(x1_diff_list_road)})
                print(f"x1_diff_list_lane Afterrrrr sotring" , {str(x1_diff_list_lane)})

                try:
                    x1_diff = x1_diff_list_road[0] - x1_diff_list_lane[0]
                    x1_diff = abs(x1_diff)

                    cv2.putText(annotated_frame, f"Distance is {x1_diff}", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                    cv2.line(annotated_frame, (x1_diff_list_road[0], 2000) , (x1_diff_list_lane[0], 2000),  (255, 0, 255), 2)

                    # x1_diff_list_road.sort(reverse=True)
                    # x1_diff_list_lane.sort(reverse=True)
                    # x2_diff_list_road.sort(reverse=True)
                    # x2_diff_list_lane.sort(reverse=True)

                    pavementWidth = x1_diff_list_lane[1]-x1_diff_list_lane[0]
                    cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4) 
                    # cv2.line(annotated_frame, (x1_diff_list_road[0], y1_diff_list_road[0]+250) , (x1_diff_list_lane[0], y1_diff_list_lane[0]+250),  (255, 0, 255), 2) 
                    # cv2.line(annotated_frame, (x1_diff_list_lane[0], y1_diff_list_lane[0]+70) , (x2_diff_list_lane[0], y2_diff_list_lane[0]+70),  (0, 0, 255), 4)
                     
                    cv2.line(annotated_frame, (x1_diff_list_lane[0], 2000) , (x2_diff_list_lane[-1], 2000),  (255, 0, 100), 4)
                except:
                    x1_diff = pavementWidth =0
                    cv2.putText(annotated_frame, f"Distance is {x1_diff}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                    cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)      
                    #cv2.line(annotated_frame, (x_diff_list_road[0], y_diff_list_road[0]-50) , (x_diff_list_lane[0], y_diff_list_lane[0]-50),  (255, 0, 255), 2)      


                # output_video.write(annotated_frame)
                endTime = time.time()
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

        except Exception as e:
            # print(f"e {E['c1'].iloc[frameN]}")
            continue
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

    dataFrame.to_csv('inner_demo_5d74a272-d4fb-4cc8-a02b-3ab23e50ecd0.csv')
    # # Write frame processing times to a CSV file
    # with open('frame_processing_times_demo.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Frame Number', 'Processing Time (seconds)'])
    #     for i, time in enumerate(frame_processing_timeList):
    #         writer.writerow([i+1, time])


    