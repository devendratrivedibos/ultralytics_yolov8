from ultralytics import YOLO
import numpy as np
import cv2
import pdb
model_seg = YOLO('shoulderWidth/29julyy3/weights/best.pt')                

if __name__ == '__main__':
    sourceseg = r'Z:/80f16f89-14e1-494c-b428-314829853b04/5b4bfa13-8e8c-42f8-bf10-9ab49833c139/pcams/pcam-0000945-RL.jpg'
    sourceseg = cv2.imread(sourceseg)
    # sourceseg = sourceseg.copy()
    height, width = sourceseg.shape[:2]
    end_col =  width // 2
    # sourceseg[0:end_col, :] = [0, 0, 0]  # RGB values for 
    
    sourceseg[1500:, 0:500] = [0, 0, 0]  # RGB values for black
    sourceseg[:, end_col:] = [0, 0, 0]  # RGB values for black
    results_segm = model_seg(sourceseg,imgsz=640, conf=0.3,iou= 0.3,device =0,show_boxes = True)  # generator of Results objects 

    annotated_frame = results_segm[0].plot(boxes=True)
    x1_diff_list_road = []
    y1_diff_list_road = []
    x2_diff_list_road = []
    y2_diff_list_road = []

    x1_diff_list_lane = []
    y1_diff_list_lane = []
    x2_diff_list_lane = []
    y2_diff_list_lane = []
    road_mark =[]
    road_points =[]
    lane_mark = []
    lane_points =[]
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

                points = mask.xy.copy()
                points = [np.asarray(point) for point in points]
                points = [np.round(point).astype(np.int32) for point in points]
                point = points[0]
                point = point.reshape(-1, 2)
                sorted_points = point[np.lexsort((point[:, 1], point[:, 0]))]
                sorted_points = sorted_points[0:20]
                if class_id == 1:
                    x1_diff_list_road.append(xs1)
                    y1_diff_list_road.append(ys1)
                    x2_diff_list_road.append(xs2)
                    y2_diff_list_road.append(ys2)
                    road_mark.append(sorted_points)
                    road_points.append(point)
                if class_id == 0:
                    x1_diff_list_lane.append(xs1)
                    y1_diff_list_lane.append(ys1)
                    x2_diff_list_lane.append(xs2)
                    y2_diff_list_lane.append(ys2)
                    lane_mark.append(sorted_points)
                    lane_points.append(point)
                # pdb.set_trace()
                for p in points[0]:
                    cv2.circle(annotated_frame, (int(p[0]),int(p[1])), 2, (255, 255, 255), -1)
                rect = cv2.minAreaRect(np.concatenate(points, axis=0))
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                center_point = np.mean(box, axis=0, dtype=np.intp)
                x_boxC , y_boxC = center_point  
                cv2.circle(annotated_frame, center_point, 5, (255, 0, 255), -1)
                for contour in points:
                    convexHull = cv2.convexHull(contour)
                    cv2.drawContours(annotated_frame, [convexHull], -1, (255, 255, 255), 3) 
            print(f"x1_diff_list_road before sotring" , {str(x1_diff_list_road)})
            print(f"x2_diff_list_road before sotring" , {str(x2_diff_list_road)})
            print(f"x1_diff_list_lane before sotring" , {str(x1_diff_list_lane)})
            print(f"x2_diff_list_lane before sotring" , {str(x2_diff_list_lane)})
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
            print(f"x1_diff_list_road afterrrrrr sotring" , {str(x1_diff_list_road)})
            print(f"x2_diff_list_road afterrrrrr sotring" , {str(x2_diff_list_road)})
            print(f"x1_diff_list_lane afterrrrrr sotring" , {str(x1_diff_list_lane)})
            print(f"x2_diff_list_lane afterrrrrr sotring" , {str(x2_diff_list_lane)})
            try:
                x1_diff = x1_diff_list_road[0] - x1_diff_list_lane[0]
                x1_diff = abs(x1_diff)
                max_index = y2_diff_list_lane.index(max(y2_diff_list_lane))
                cv2.putText(annotated_frame, f"Distance is {x1_diff}", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                cv2.line(annotated_frame, (x1_diff_list_road[0], 2000) , (x1_diff_list_lane[0], 2000),  (255, 0, 255), 2)
                pavementWidth = abs(x1_diff_list_lane[-2]-x2_diff_list_lane[-1])
                cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4) 
                cv2.line(annotated_frame, (x1_diff_list_lane[-2], 1800) , (x2_diff_list_lane[-1], 1800),  (100, 200, 100), 4)
            except Exception as e:
                print(e)
                x1_diff = pavementWidth =0
                # pavementWidth = x1_diff_list_lane[-1]-x1_diff_list_lane[-2]
                cv2.putText(annotated_frame, f"Distance is {x1_diff}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4) 
                cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)      
                #cv2.line(annotated_frame, (x_diff_list_road[0], y_diff_list_road[0]-50) , (x_diff_list_lane[0], y_diff_list_lane[0]-50),  (255, 0, 255), 2)      

    # cv2.imwrite('merged_segs009ll.jpg', annotated_frame)
    cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    cv2.imshow('im', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
