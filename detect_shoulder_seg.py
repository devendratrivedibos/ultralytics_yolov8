from ultralytics import YOLO
import numpy as np
import cv2
import pdb
model_seg = YOLO('shoulderWidth/65mar/weights/best.pt')                

if __name__ == '__main__':

    sourceseg = r'F:/5a471b1f-ecbc-4e60-a7ba-8416627c6c9c/d7b2e376-53af-45b6-8dc7-a02c38d448c8/pcams/pcam-0000451-LL.jpg'
    # sourceseg = cv2.imread(sourceseg)
    # sourceseg = sourceseg.copy()
    # height, width = sourceseg.shape[:2]
    # start_col = 0
    # end_col = height // 2
    # sourceseg[0:end_col, :] = [0, 0, 0]  # RGB values for black
    results_segm = model_seg(sourceseg,imgsz=640, conf=0.4,iou= 0.5,device =0,show_boxes = True,save=True)  # generator of Results objects 

    annotated_frame = results_segm[0].plot(boxes=True)
    x1_diff_list_road = []
    y1_diff_list_road = []
    x2_diff_list_road = []
    y2_diff_list_road = []

    x1_diff_list_lane = []
    y1_diff_list_lane = []
    x2_diff_list_lane = []
    y2_diff_list_lane = []

    # pdb.set_trace()
    for result_seg in results_segm:
        if result_seg.masks is not None:
            for mask, box in zip(result_seg.masks, result_seg.boxes):
             
                class_id = int(box.cls[0])  # Access to the class ID
                confidence = box.conf[0]    # Access to the confidence
                class_name = result_seg.names[class_id]  # Class name based on the ID
                label = f"{class_name}: {confidence:.2f}"
                box_coords_xyxy = box.xyxy.cpu().numpy()         ###CORDINATES BOX
                box_coords_xywh = box.xywh.cpu().numpy()

                xs1, ys1,xs2,ys2 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1]),int(box_coords_xyxy[0][2]),int(box_coords_xyxy[0][3])
                xsc, ysc = int(box_coords_xywh[0][0]), int(box_coords_xywh[0][1])
                box_center_label =f'Centre pixel box cordinates are  {str(xsc)}  {str(ysc)}  for {class_name}'
                print(box_center_label)

                if class_id == 1:
                    x1_diff_list_road.append(xs1)
                    y1_diff_list_road.append(ys2)
                    x2_diff_list_road.append(xs1)
                    y2_diff_list_road.append(ys2)
                if class_id == 0:
                    x1_diff_list_lane.append(xs1)
                    y1_diff_list_lane.append(ys2)
                    x2_diff_list_lane.append(xs1)
                    y2_diff_list_lane.append(ys2)
                points = mask.xy.copy()
                # points = mask.xy.cpu().numpy()
                points = [np.asarray(point) for point in points]
                points = [np.round(point).astype(np.int32) for point in points]
                
                for p in points[0]:
                    cv2.circle(annotated_frame, (int(p[0]),int(p[1])), 2, (255, 255, 255), -1)
                # points = np.round(points).astype(np.int32)
                # points_for_cv2 = points[:, np.newaxis, :]  

                rect = cv2.minAreaRect(np.concatenate(points, axis=0))
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                center_point = np.mean(box, axis=0, dtype=np.intp)
                x_boxC , y_boxC = center_point

                box_center_label =f'{class_name} Centre pixel segment cordinates are  {str(x_boxC)}  {str(y_boxC)}'
                print(box_center_label)    
                cv2.circle(annotated_frame, center_point, 5, (255, 0, 255), -1)


                for contour in points:
                    convexHull = cv2.convexHull(contour)
                    cv2.drawContours(annotated_frame, [convexHull], -1, (255, 255, 255), 3) 

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



                pavementWidth = x1_diff_list_lane[1]-x1_diff_list_lane[0]
                cv2.putText(annotated_frame, f"Distance is {x1_diff}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                cv2.line(annotated_frame, (x1_diff_list_road[0], y1_diff_list_road[0]-50) , (x1_diff_list_lane[0], y1_diff_list_lane[0]-50),  (255, 0, 255), 2) 
                cv2.line(annotated_frame, (x1_diff_list_lane[0], y1_diff_list_lane[0]) , (x2_diff_list_lane[0], y2_diff_list_lane[0]),  (255, 0, 255), 2)
            except:
                x1_diff = 0
                cv2.putText(annotated_frame, f"Distance is {x1_diff}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                #cv2.line(annotated_frame, (x_diff_list_road[0], y_diff_list_road[0]-50) , (x_diff_list_lane[0], y_diff_list_lane[0]-50),  (255, 0, 255), 2)                 
    cv2.imwrite('E:/Devendra_Files/ultralytics-main/ultralytics-main/merged_segs009ll.jpg', annotated_frame)
