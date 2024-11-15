from ultralytics import YOLO
import cv2
from glob import glob
import os
import time
import pandas as pd
try:
    from transformers.inventory.gps_conversion import GPS_Cord
except:
    from gps_conversion import GPS_Cord

from egestors.report.sheets.sheet import Sheet


class InventoryFeatures_Inner:
    def __init__(self, video_path , gps_data, output_csv, sectionID):
        """
        Initializes the InventoryFeatures_Inner class with the given video path, GPS data, output CSV path, and section ID.
        
        Args:
            video_path (str): Path to the video or image directory.
            gps_data (DataFrame): GPS data for the video.
            output_csv (str): Path to save the output CSV file.
            sectionID (str): Section ID to be used in the output.
        """
        self.video_path = video_path
        self.gps_data = gps_data
        self.output_csv = output_csv
        parts = self.video_path.split(os.sep)[0]
        parts = parts.replace("pcams", "")
        print(self.output_csv)
        self.output_csv_ = self.generate_new_csv_path()
        self.sectionID = sectionID
        self.total_processing_time = 0
        self.frame_processing_timeList = []
        self.feature_data = {
            'Section_ID': [],
            'RegionIndex': [],
            'data_type': [],
            'annotation_type': [],
            'Feature Type': [],
            'PointsX1': [],
            'PointsY1': [],
            'PointsX2': [],
            'PointsY2': [],
            'PointsX3': [],
            'PointsY3': [],
            'PointsX4': [],
            'PointsY4': [],
            'Latitude': [],
            'Longitude': [],
            'Altitude': [],
            'confidence': []
        }
        self.feature_data_widt = {'Section_ID': [], 'RegionIndex': [], 'shoulderWidth': [], 'pavementWidth': []}
        self.frameN = 0
        self.script_dir = os.path.dirname(__file__)
        self.model_segm_soulder = YOLO(
            os.path.normpath(os.path.join(self.script_dir, '..', '..', 'ai_model', 'soulderWidt.pt')))
        self.model_det = YOLO(
            os.path.normpath(os.path.join(self.script_dir, '..', '..', 'ai_model', '16mar_RF-wright.pt')))
        self.model_seg = YOLO(
            os.path.normpath(os.path.join(self.script_dir, '..', '..', 'ai_model', '16mar_segmentation.pt')))

    def generate_new_csv_path(self):
        """
        Generates a new path for the output CSV by adding a 'soulder_' prefix to the file name.
        
        Returns:
            str: The new output CSV path with the updated file name.
        """
        path_parts = self.output_csv.rsplit('\\', 1)  # Split path into directory and file name
        file_name = path_parts[1]
        new_file_name = f"soulder_{file_name}"  # Add 'soulder_' prefix to the file name
        new_path = f"{path_parts[0]}\\{new_file_name}"  # Combine the directory with the new file name
        return new_path

    def sort_coordinate_lists(self, x_list, y_list):
        """
        Sorts the x and y coordinate lists based on the x values.
        
        Args:
            x_list (list): List of x coordinates.
            y_list (list): List of y coordinates.
        
        Returns:
            tuple: Sorted x and y lists.
        """
        combined = list(zip(x_list, y_list))
        combined = sorted(combined, key=lambda pair: pair[0])
        x_list, y_list = zip(*combined)
        return list(x_list), list(y_list)

    def mask_frame(self, frame):
        """
        Masks a portion of the input frame by making a part of it black.        
        Args:
            frame (numpy array): The input video frame.
        Returns:
            numpy array: The masked frame.
        """
        height, width = frame.shape[:2]
        mask_frame = frame.copy()
        start_col = 0
        end_col = width // 5
        height = int(height - height // 2.5)
        mask_frame[0:height, start_col:end_col] = [0, 0, 0]  # RGB values for black
        return mask_frame

    def unmask_frame(self, frame, annotated_frame):
        """
        Unmasks the frame by restoring the original part of the frame that was masked.
        Args:
            frame (numpy array): The original frame.
            annotated_frame (numpy array): The annotated frame to be unmasked.
        Returns:
            numpy array: The unmasked frame.
        """
        height, width = frame.shape[:2]
        start_col = 0
        end_col = width // 5
        height = int(height - height // 2.5)
        annotated_frame[0:height, start_col:end_col] = frame[0:height, start_col:end_col]
        return annotated_frame

    def annotate_frame(self, annotated_frame, results_det, results_segm):
        """
        Annotates the frame with detection and segmentation results by drawing rectangles and adding text.
        
        Args:
            annotated_frame (numpy array): The frame to be annotated.
            results_det (list): Detection results.
            results_segm (list): Segmentation results.
        """
        for result_det in results_det:
            for box in result_det.boxes:
                xd, yd, wd, hd = box.xywh.cpu().numpy()[0]
                xd, yd, wd, hd = int(xd), int(yd), int(wd), int(hd)
                if wd < 80 and hd < 80:
                    continue
                xd1, yd1, xd4, yd4 = box.xyxy.cpu().numpy()[0]
                xd1, yd1, xd4, yd4 = int(xd1), int(yd1), int(xd4), int(yd4)
                if xd1 < 1200:
                    continue
                box_center_label = f'Road Furniture Centre coordinates are {str(xd)} {str(yd)}'
                class_id_det = int(box.cls.cpu().numpy())
                class_name_det = result_det.names[int(box.cls)]
                confidence_det = float(box.conf.cpu())
                label_det = f"{class_name_det}: {confidence_det:.2f}"
                cv2.putText(annotated_frame, label_det, (xd1, yd1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
                cv2.putText(annotated_frame, box_center_label, (1500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                            4)
                cv2.rectangle(annotated_frame, (xd1, yd1), (xd4, yd4), (120, 0, 255), 4)

        for result_seg in results_segm:
            if result_seg.masks is not None:
                for mask, box in zip(result_seg.masks, result_seg.boxes):
                    xs, ys, ws, hs = box.xywh.cpu().numpy()[0]
                    xs, ys, ws, hs = int(xs), int(ys), int(ws), int(hs)
                    xs1, ys1, xs4, ys4 = box.xyxy.cpu().numpy()[0]
                    xs1, ys1, xs4, ys4 = int(xs1), int(ys1), int(xs4), int(ys4)
                    box_center_label = f'Segmentation Centre coordinates are {str(xs)} {str(ys)}'
                    class_id_seg = int(box.cls.cpu().numpy())
                    class_name_seg = result_seg.names[int(box.cls)]
                    confidence_seg = float(box.conf.cpu())
                    label_seg = f"{class_name_seg}: {confidence_seg:.2f}"
                    cv2.putText(annotated_frame, label_seg, (xs1, ys1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                4)
                    cv2.putText(annotated_frame, box_center_label, (1500, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 4)
                    # cv2.rectangle(annotated_frame, (xs1, ys1), (xs4, ys4), (0, 255, 0), 4)

    def calculate_width(self, frame, results_segm, frameN):
        """
        Calculates the pavement width from the segmentation results and annotates the frame with the width.
        Args:
            frame (numpy array): The input video frame.
            results_segm (list): Segmentation results.
            frameN (int): Frame number.
        Returns:
            numpy array: The frame annotated with the calculated width.
        """
        x1_diff_list_road = []
        y1_diff_list_road = []
        x2_diff_list_road = []
        y2_diff_list_road = []
        x1_diff_list_lane = []
        y1_diff_list_lane = []
        x2_diff_list_lane = []
        y2_diff_list_lane = []
        leftcordinates = []
        rightcordinates = []
        annotated_frame = frame.copy()
        for result_seg in results_segm:
            if result_seg.masks is not None:
                for mask, box in zip(result_seg.masks, result_seg.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0].cpu())
                    class_name = result_seg.names[class_id]

                    box_coords_xyxy = box.xyxy.cpu().numpy()
                    xs, ys, ws, hs = box.xywh.cpu().numpy()[0]
                    xs, ys, ws, hs = int(xs), int(ys), int(ws), int(hs)

                    if ws < 200 or hs < 200:
                        continue
                    print(ws, hs)
                    xs1, ys1, xs2, ys2 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1]), int(
                        box_coords_xyxy[0][2]), int(box_coords_xyxy[0][3])

                    # Append coordinates to the respective lists
                    if class_id == 1:
                        x1_diff_list_road.append(xs1)
                        y1_diff_list_road.append(ys1)
                        x2_diff_list_road.append(xs2)
                        y2_diff_list_road.append(ys2)
                    elif class_id == 0:
                        x1_diff_list_lane.append(xs1)
                        y1_diff_list_lane.append(ys1)
                        x2_diff_list_lane.append(xs2)
                        y2_diff_list_lane.append(ys2)

        # # Sorting the coordinate lists
        # x1_diff_list_road.sort()
        # x1_diff_list_lane.sort()
        # x2_diff_list_road.sort()
        # x2_diff_list_lane.sort()
        try:
            x1_diff_list_lane, y1_diff_list_lane = self.sort_coordinate_lists(x1_diff_list_lane, y1_diff_list_lane)
            x2_diff_list_lane, y2_diff_list_lane = self.sort_coordinate_lists(x2_diff_list_lane, y2_diff_list_lane)
            x1_diff_list_road, y1_diff_list_road = self.sort_coordinate_lists(x1_diff_list_road, y1_diff_list_road)
            x2_diff_list_road, y2_diff_list_road = self.sort_coordinate_lists(x2_diff_list_road, y2_diff_list_road)

            leftcordinates = (x1_diff_list_lane[0], y2_diff_list_lane[0])
            rightcordinates = (x2_diff_list_lane[-1], y2_diff_list_lane[0])  # y2_diff_list_lane[-1]
            print(leftcordinates, rightcordinates)
            pavementWidth = Sheet.calcLength(leftcordinates, rightcordinates)
            cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 4)
            # cv2.line(annotated_frame, (x1_diff_list_lane[0], 1800), (x2_diff_list_lane[-1], 1800), (100, 200, 100), 4)
            cv2.line(annotated_frame, leftcordinates, rightcordinates, (100, 200, 100), 4)


        except Exception as e:
            print(e)
            x1_diff = [0]
            pavementWidth = [0]
            cv2.putText(annotated_frame, f"Distance is {x1_diff}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                        4)
            cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 4)
        self.feature_data_widt['Section_ID'].append(self.sectionID)
        self.feature_data_widt['RegionIndex'].append(frameN)
        self.feature_data_widt['shoulderWidth'].append(0)
        self.feature_data_widt['pavementWidth'].append(pavementWidth[0])
        return annotated_frame

    def process_frame(self, frame, frameN):
        """
        Processes a single frame for segmentation, detection, and width calculation.
        Args:
            frame (numpy array): The video frame to be processed.
            frameN (int): The frame number.
        Returns:
            tuple: Annotated frame, segmentation results, and detection results.
        """
        mask_frame = self.mask_frame(frame)
        results_segm = self.model_seg(mask_frame, conf=0.4, iou=0.47, imgsz=640, device=0, classes=[0, 1],
                                      show_boxes=False)
        results_segm_soulder = self.model_segm_soulder(mask_frame, conf=0.4, iou=0.47, imgsz=640, device=0,
                                                       show_boxes=False)  ####new added
        results_det = self.model_det(mask_frame, conf=0.40, imgsz=1024, device=0)
        annotated_frame = results_segm_soulder[0].plot(boxes=False)
        annotated_frame1 = self.unmask_frame(frame, annotated_frame)
        self.annotate_frame(annotated_frame1, results_det, results_segm)
        annotated_frame1 = self.calculate_width(annotated_frame, results_segm_soulder, frameN)  ####new added
        return annotated_frame1, results_segm, results_det

    def extract_features(self, results_segm, results_det, frame, frameN):
        """
        Extracts features from the segmentation and detection results of a frame.
        Args:
            results_segm (list): Segmentation results.
            results_det (list): Detection results.
            frame (numpy array): The video frame.
            frameN (int): The frame number.
        """
        start_col = 0
        end_col = frame.shape[1] // 3
        for result_seg in results_segm:
            if result_seg.masks is not None:
                for mask, box in zip(result_seg.masks, result_seg.boxes):
                    self.process_segmentation_result(mask, box, frame, frameN, result_seg)
        for result_det in results_det:
            for box in result_det.boxes:
                self.process_detection_result(box, frame, frameN, result_det)

    def process_segmentation_result(self, mask, box, frame, frameN, result_seg):
        class_id = int(box.cls[0])

        confidence = float(box.conf[0].cpu())
        class_name = result_seg.names[class_id]
        box_coords_xyxy = box.xyxy.cpu().numpy()
        box_coords_xywh = box.xywh.cpu().numpy()
        xs1, ys1 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1])
        xsc, ysc = int(box_coords_xywh[0][0]), int(box_coords_xywh[0][1])
        gps_loc = self.gps_data.iloc[frameN]

        self.append_feature_data(class_id, class_name, confidence, xsc, ysc, gps_loc, frameN, box)

    def process_detection_result(self, box, frame, frameN, result_det):
        xd, yd, wd, hd = box.xywh.cpu().numpy()[0]

        xd, yd = int(xd), int(yd)
        class_id = int(box.cls.cpu().numpy())
        class_name = result_det.names[class_id]
        confidence = float(box.conf.cpu())
        gps_loc = self.gps_data.iloc[frameN]

        self.append_feature_data(class_id, class_name, confidence, xd, yd, gps_loc, frameN, box)

    def append_feature_data(self, class_id, class_name, confidence, x, y, gps_loc, frameN, box):
        self.feature_data['Section_ID'].append(self.sectionID)
        self.feature_data['RegionIndex'].append(frameN)
        self.feature_data['data_type'].append('00')
        self.feature_data['annotation_type'].append(class_id)
        self.feature_data['Feature Type'].append(class_name)
        xd1, yd1, xd4, yd4 = (box.xyxy.cpu().numpy()[0])
        xd1, yd1, xd4, yd4 = int(xd1), int(yd1), int(xd4), int(yd4)
        xd3 = xd1
        xd2 = xd4
        yd3 = yd4
        yd2 = yd1
        self.feature_data['PointsX1'].append(xd1)
        self.feature_data['PointsY1'].append(yd1)
        self.feature_data['PointsX2'].append(xd2)
        self.feature_data['PointsY2'].append(yd2)
        self.feature_data['PointsX3'].append(xd3)
        self.feature_data['PointsY3'].append(yd3)
        self.feature_data['PointsX4'].append(xd4)
        self.feature_data['PointsY4'].append(yd4)
        self.feature_data['Latitude'].append(gps_loc['lat'])
        self.feature_data['Longitude'].append(gps_loc['long'])
        self.feature_data['Altitude'].append(gps_loc['hell'])
        self.feature_data['confidence'].append(confidence)

    def save_features_to_csv(self):
        """ Saves the extracted features to a CSV file."""
        dataFrame = pd.DataFrame(self.feature_data)
        dataFrame.to_csv(self.output_csv, index=False)
        print("Data saved to CSV:", self.output_csv)
        dataFrame = pd.DataFrame(self.feature_data_widt)
        dataFrame.to_csv(self.output_csv_, index=False)
        print("Data saved to CSV:", self.output_csv_)

    def process_video(self):
        image_dir = self.video_path
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))  # Adjust the extension if needed
        self.image_paths = sorted([p for p in glob(os.path.join(image_dir, "*.jpg")) if "RL" in p])
        # cv2.namedWindow("Road Furniture Detection", cv2.WINDOW_NORMAL)
        i = 0
        for image_path in self.image_paths:
            frame = cv2.imread(image_path)
            print(f'Invertory process:{i}', flush=True)
            i += 1
            startTime = time.time()
            annotated_frame1, results_segm, results_det = self.process_frame(frame, self.frameN)
            self.extract_features(results_segm, results_det, frame, self.frameN)
            endTime = time.time()
            frameProcessTime = endTime - startTime
            print(f"Processed {image_path} in {frameProcessTime:.4f} seconds")
            self.total_processing_time += frameProcessTime
            self.frame_processing_timeList.append(frameProcessTime)
            self.frameN += 1
            # cv2.imshow("Road Furniture Detection", annotated_frame1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # cap.release()
        cv2.destroyAllWindows()
        self.save_features_to_csv()


class InventoryFeatures_Outer:
    def __init__(self, video_path, gps_data, output_csv, sectionID):
        """
        Initializes the InventoryFeatures_Outer class with the given video path, GPS data, output CSV path, and section ID.
        
        Args:
            video_path (str): Path to the video or image directory.
            gps_data (DataFrame): GPS data for the video.
            output_csv (str): Path to save the output CSV file.
            sectionID (str): Section ID to be used in the output.
        """
        self.video_path = video_path
        self.gps_data = gps_data
        self.output_csv = output_csv
        parts = self.video_path.split(os.sep)[0]
        parts = parts.replace("pcams", "")
        print(self.output_csv)
        self.output_csv_ = self.generate_new_csv_path()
        self.sectionID = sectionID
        self.total_processing_time = 0
        self.frame_processing_timeList = []
        self.feature_data = {
            'Section_ID': [],
            'RegionIndex': [],
            'data_type': [],
            'annotation_type': [],
            'Feature Type': [],
            'PointsX1': [],
            'PointsY1': [],
            'PointsX2': [],
            'PointsY2': [],
            'PointsX3': [],
            'PointsY3': [],
            'PointsX4': [],
            'PointsY4': [],
            'Latitude': [],
            'Longitude': [],
            'Altitude': [],
            'confidence': []
        }

        self.feature_data_widt = {'Section_ID': [], 'RegionIndex': [], 'shoulderWidth': [], 'pavementWidth': []}
        self.frameN = 0
        self.script_dir = os.path.dirname(__file__)
        self.model_det = YOLO(
            os.path.normpath(os.path.join(self.script_dir, '..', '..', 'ai_model', '16mar_RF-wright.pt')))
        self.model_seg = YOLO(
            os.path.normpath(os.path.join(self.script_dir, '..', '..', 'ai_model', '16mar_segmentation.pt')))
        self.model_segm_soulder = YOLO(
            os.path.normpath(os.path.join(self.script_dir, '..', '..', 'ai_model', 'soulderWidt.pt')))

    def generate_new_csv_path(self):
        """
        Generates a new path for the output CSV by adding a 'soulder_' prefix to the file name.
        
        Returns:
            str: The new output CSV path with the updated file name.
        """
        path_parts = self.output_csv.rsplit('\\', 1)  # Split path into directory and file name
        file_name = path_parts[1]
        new_file_name = f"soulder_{file_name}"  # Add 'soulder_' prefix to the file name
        new_path = f"{path_parts[0]}\\{new_file_name}"  # Combine the directory with the new file name
        return new_path

    def sort_coordinate_lists(self, x_list, y_list):
        """
        Sorts the x and y coordinate lists based on the x values.
        
        Args:
            x_list (list): List of x coordinates.
            y_list (list): List of y coordinates.
        
        Returns:
            tuple: Sorted x and y lists.
        """
        combined = list(zip(x_list, y_list))
        combined = sorted(combined, key=lambda pair: pair[0])
        x_list, y_list = zip(*combined)
        return list(x_list), list(y_list)

    def mask_frame(self, frame):
        """
        Masks a portion of the input frame by making a part of it black.        
        Args:
            frame (numpy array): The input video frame.
        Returns:
            numpy array: The masked frame.
        """
        height, width = frame.shape[:2]
        mask_frame = frame.copy()
        start_col = 0
        end_col = width - (width // 3)
        mask_frame[:, end_col:width] = [0, 0, 0]
        endrow = height // 3
        # mask_frame[:endrow,:] = [0, 0, 0]
        mask_frame[1350:, :] = [0, 0, 0]
        return mask_frame

    def unmask_frame(self, frame, annotated_frame):
        """
        Unmasks the frame by restoring the original part of the frame that was masked.
        Args:
            frame (numpy array): The original frame.
            annotated_frame (numpy array): The annotated frame to be unmasked.
        Returns:
            numpy array: The unmasked frame.
        """
        height, width = frame.shape[:2]
        start_col = 0
        # end_col = width // 3
        end_col = width - (width // 3)
        annotated_frame[:, end_col:width] = frame[:, end_col:width]
        endrow = height // 3
        # annotated_frame[:endrow,:] = frame[:endrow,:] 
        annotated_frame[1350:, :] = frame[1350:, :]
        return annotated_frame

    def annotate_frame(self, annotated_frame, results_det, results_segm):
        """
        Annotates the frame with detection and segmentation results by drawing rectangles and adding text.
        
        Args:
            annotated_frame (numpy array): The frame to be annotated.
            results_det (list): Detection results.
            results_segm (list): Segmentation results.
        """
        for result_det in results_det:
            for box in result_det.boxes:
                xd, yd, wd, hd = box.xywh.cpu().numpy()[0]
                xd, yd, wd, hd = int(xd), int(yd), int(wd), int(hd)
                if wd < 80 and hd < 80:
                    continue
                xd1, yd1, xd4, yd4 = box.xyxy.cpu().numpy()[0]
                xd1, yd1, xd4, yd4 = int(xd1), int(yd1), int(xd4), int(yd4)
                if xd1 > 1000:
                    continue
                box_center_label = f'Road Furniture Centre coordinates are {str(xd)} {str(yd)}'
                class_id_det = int(box.cls.cpu().numpy())
                class_name_det = result_det.names[int(box.cls)]
                confidence_det = float(box.conf.cpu())
                label_det = f"{class_name_det}: {confidence_det:.2f}"
                cv2.putText(annotated_frame, label_det, (xd1, yd1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 4)
                cv2.putText(annotated_frame, box_center_label, (1500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                            4)
                cv2.rectangle(annotated_frame, (xd1, yd1), (xd4, yd4), (120, 0, 255), 4)

        for result_seg in results_segm:
            if result_seg.masks is not None:
                for mask, box in zip(result_seg.masks, result_seg.boxes):
                    xs, ys, ws, hs = box.xywh.cpu().numpy()[0]
                    xs, ys, ws, hs = int(xs), int(ys), int(ws), int(hs)
                    xs1, ys1, xs4, ys4 = box.xyxy.cpu().numpy()[0]
                    xs1, ys1, xs4, ys4 = int(xs1), int(ys1), int(xs4), int(ys4)
                    box_center_label = f'Segmentation Centre coordinates are {str(xs)} {str(ys)}'
                    class_id_seg = int(box.cls.cpu().numpy())
                    class_name_seg = result_seg.names[int(box.cls)]
                    confidence_seg = float(box.conf.cpu())
                    label_seg = f"{class_name_seg}: {confidence_seg:.2f}"
                    cv2.putText(annotated_frame, label_seg, (xs1, ys1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                4)
                    cv2.putText(annotated_frame, box_center_label, (1500, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 4)
                    # cv2.rectangle(annotated_frame, (xs1, ys1), (xs4, ys4), (0, 255, 0), 4)

    def calculate_width(self, frame, results_segm, frameN):
        """
        Calculates the pavement width from the segmentation results and annotates the frame with the width.
        Args:
            frame (numpy array): The input video frame.
            results_segm (list): Segmentation results.
            frameN (int): Frame number.
        Returns:
            numpy array: The frame annotated with the calculated width.
        """
        x1_diff_list_road = []
        y1_diff_list_road = []
        x2_diff_list_road = []
        y2_diff_list_road = []

        x1_diff_list_lane = []
        y1_diff_list_lane = []
        x2_diff_list_lane = []
        y2_diff_list_lane = []
        leftcordinates = []
        rightcordinates = []
        annotated_frame = frame.copy()
        for result_seg in results_segm:
            if result_seg.masks is not None:
                for mask, box in zip(result_seg.masks, result_seg.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0].cpu())
                    class_name = result_seg.names[class_id]

                    box_coords_xyxy = box.xyxy.cpu().numpy()
                    xs1, ys1, xs2, ys2 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1]), int(
                        box_coords_xyxy[0][2]), int(box_coords_xyxy[0][3])

                    # Append coordinates to the respective lists
                    if class_id == 1:
                        x1_diff_list_road.append(xs1)
                        y1_diff_list_road.append(ys1)
                        x2_diff_list_road.append(xs2)
                        y2_diff_list_road.append(ys2)
                    elif class_id == 0:
                        x1_diff_list_lane.append(xs1)
                        y1_diff_list_lane.append(ys1)
                        x2_diff_list_lane.append(xs2)
                        y2_diff_list_lane.append(ys2)

        # Sorting the coordinate lists
        # x1_diff_list_road.sort()
        # x1_diff_list_lane.sort()
        # x2_diff_list_road.sort()
        # x2_diff_list_lane.sort()
        try:
            x1_diff_list_lane, y1_diff_list_lane = self.sort_coordinate_lists(x1_diff_list_lane, y1_diff_list_lane)
            x2_diff_list_lane, y2_diff_list_lane = self.sort_coordinate_lists(x2_diff_list_lane, y2_diff_list_lane)
            x1_diff_list_road, y1_diff_list_road = self.sort_coordinate_lists(x1_diff_list_road, y1_diff_list_road)
            x2_diff_list_road, y2_diff_list_road = self.sort_coordinate_lists(x2_diff_list_road, y2_diff_list_road)

            leftcordinates = (0, y2_diff_list_lane[0])
            rightcordinates = (x1_diff_list_lane[0], y2_diff_list_lane[0])
            x1_diff = Sheet.calcLength(leftcordinates, rightcordinates)
            pavementWidth = abs(x1_diff_list_lane[0] - x2_diff_list_lane[-1])
            leftcordinates = (x1_diff_list_lane[0], y2_diff_list_lane[0])
            rightcordinates = (x2_diff_list_lane[-1], y2_diff_list_lane[0])  # y2_diff_list_lane[-1]
            print(leftcordinates, rightcordinates)
            pavementWidth = Sheet.calcLength(leftcordinates, rightcordinates)
            # Annotating the frame with distance information
            cv2.putText(annotated_frame, f"Soulder Distance is {x1_diff}", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 4)
            cv2.line(annotated_frame, (x1_diff_list_road[0], 2000), (x1_diff_list_lane[0], 2000), (255, 0, 255), 2)
            cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 4)
            # cv2.line(annotated_frame, (x1_diff_list_lane[0], 1800), (x2_diff_list_lane[-1], 1800), (100, 200, 100), 4)
            cv2.line(annotated_frame, leftcordinates, rightcordinates, (100, 200, 100), 4)

        except Exception as e:
            print(e)
            x1_diff = [0]
            pavementWidth = [0]
            cv2.putText(annotated_frame, f"Distance is {x1_diff}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                        4)
            cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 4)
        self.feature_data_widt['Section_ID'].append(self.sectionID)
        self.feature_data_widt['RegionIndex'].append(frameN)
        self.feature_data_widt['shoulderWidth'].append(x1_diff[0])
        self.feature_data_widt['pavementWidth'].append(pavementWidth[0])
        return annotated_frame

    def process_frame(self, frame, frameN):
        """
        Processes a single frame for segmentation, detection, and width calculation.
        Args:
            frame (numpy array): The video frame to be processed.
            frameN (int): The frame number.
        Returns:
            tuple: Annotated frame, segmentation results, and detection results.
        """
        mask_frame = self.mask_frame(frame)
        results_segm = self.model_seg(mask_frame, conf=0.4, iou=0.47, imgsz=640, device=0, classes=[0, 1],
                                      show_boxes=False)
        results_det = self.model_det(mask_frame, conf=0.40, imgsz=1024, device=0)
        results_segm_soulder = self.model_segm_soulder(mask_frame, conf=0.4, iou=0.47, imgsz=640, device=0,
                                                       show_boxes=False)  ####new added
        annotated_frame = results_segm[0].plot(boxes=False)
        annotated_frame1 = self.unmask_frame(frame, annotated_frame)
        self.annotate_frame(annotated_frame1, results_det, results_segm)
        annotated_frame1 = self.calculate_width(annotated_frame, results_segm_soulder, frameN)  ####new added
        return annotated_frame1, results_segm, results_det

    def extract_features(self, results_segm, results_det, frame, frameN):
        """
        Extracts features from the segmentation and detection results of a frame.
        Args:
            results_segm (list): Segmentation results.
            results_det (list): Detection results.
            frame (numpy array): The video frame.
            frameN (int): The frame number.
        """
        start_col = 0
        end_col = frame.shape[1] // 3
        for result_seg in results_segm:
            if result_seg.masks is not None:
                for mask, box in zip(result_seg.masks, result_seg.boxes):
                    self.process_segmentation_result(mask, box, frame, frameN, result_seg)
        for result_det in results_det:
            for box in result_det.boxes:
                self.process_detection_result(box, frame, frameN, result_det)

    def process_segmentation_result(self, mask, box, frame, frameN, result_seg):
        class_id = int(box.cls[0])

        confidence = float(box.conf[0].cpu())
        class_name = result_seg.names[class_id]
        box_coords_xyxy = box.xyxy.cpu().numpy()
        box_coords_xywh = box.xywh.cpu().numpy()
        xs1, ys1 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1])
        xsc, ysc = int(box_coords_xywh[0][0]), int(box_coords_xywh[0][1])
        gps_loc = self.gps_data.iloc[frameN]

        self.append_feature_data(class_id, class_name, confidence, xsc, ysc, gps_loc, frameN, box)

    def process_detection_result(self, box, frame, frameN, result_det):
        xd, yd, wd, hd = box.xywh.cpu().numpy()[0]
        xd, yd = int(xd), int(yd)
        class_id = int(box.cls.cpu().numpy())
        class_name = result_det.names[class_id]
        confidence = float(box.conf.cpu())
        gps_loc = self.gps_data.iloc[frameN]

        self.append_feature_data(class_id, class_name, confidence, xd, yd, gps_loc, frameN, box)

    def append_feature_data(self, class_id, class_name, confidence, x, y, gps_loc, frameN, box):
        self.feature_data['Section_ID'].append(self.sectionID)
        self.feature_data['RegionIndex'].append(frameN)
        self.feature_data['data_type'].append('00')
        self.feature_data['annotation_type'].append(class_id)
        self.feature_data['Feature Type'].append(class_name)
        xd1, yd1, xd4, yd4 = (box.xyxy.cpu().numpy()[0])
        xd1, yd1, xd4, yd4 = int(xd1), int(yd1), int(xd4), int(yd4)
        xd3 = xd1
        xd2 = xd4
        yd3 = yd4
        yd2 = yd1
        self.feature_data['PointsX1'].append(xd1)
        self.feature_data['PointsY1'].append(yd1)
        self.feature_data['PointsX2'].append(xd2)
        self.feature_data['PointsY2'].append(yd2)
        self.feature_data['PointsX3'].append(xd3)
        self.feature_data['PointsY3'].append(yd3)
        self.feature_data['PointsX4'].append(xd4)
        self.feature_data['PointsY4'].append(yd4)
        self.feature_data['Latitude'].append(gps_loc['lat'])
        self.feature_data['Longitude'].append(gps_loc['long'])
        self.feature_data['Altitude'].append(gps_loc['hell'])
        self.feature_data['confidence'].append(confidence)

    def save_features_to_csv(self):
        """ Saves the extracted features to a CSV file."""
        dataFrame = pd.DataFrame(self.feature_data)
        dataFrame.to_csv(self.output_csv, index=False)
        print("Data saved to CSV:", self.output_csv)
        dataFrame = pd.DataFrame(self.feature_data_widt)
        dataFrame.to_csv(self.output_csv_, index=False)
        print("Data saved to CSV:", self.output_csv_)

    def process_video(self):
        image_dir = self.video_path
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))  # Adjust the extension if needed
        self.image_paths = sorted([p for p in glob(os.path.join(image_dir, "*.jpg")) if "RL" in p])
        # cv2.namedWindow("Road Furniture Detection", cv2.WINDOW_NORMAL)
        i = 0
        for image_path in self.image_paths:
            frame = cv2.imread(image_path)
            print(f'Invertory process:{i}', flush=True)
            i += 1
            startTime = time.time()
            annotated_frame1, results_segm, results_det = self.process_frame(frame, self.frameN)
            self.extract_features(results_segm, results_det, frame, self.frameN)
            endTime = time.time()
            frameProcessTime = endTime - startTime
            print(f"Processed {image_path} in {frameProcessTime:.4f} seconds")
            self.total_processing_time += frameProcessTime
            self.frame_processing_timeList.append(frameProcessTime)
            self.frameN += 1
            # cv2.imshow("Road Furniture Detection", annotated_frame1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # cap.release()
        cv2.destroyAllWindows()
        self.save_features_to_csv()


class InventoryFeatures_middle:
    def __init__(self, video_path, gps_data, output_csv, sectionID):
        """
        Initializes the InventoryFeatures_middle class with the given video path, GPS data, output CSV path, and section ID.
        
        Args:
            video_path (str): Path to the video or image directory.
            gps_data (DataFrame): GPS data for the video.
            output_csv (str): Path to save the output CSV file.
            sectionID (str): Section ID to be used in the output.
        """
        self.video_path = video_path
        self.gps_data = gps_data
        self.output_csv = output_csv
        parts = self.video_path.split(os.sep)[0]
        parts = parts.replace("pcams", "")
        print(self.output_csv)
        # self.output_csv_ = self.generate_new_csv_path()
        self.output_csv_ = "soulder_" + self.output_csv
        self.sectionID = sectionID
        self.total_processing_time = 0
        self.frame_processing_timeList = []
        self.feature_data = {
            'Section_ID': [],
            'RegionIndex': [],
            'data_type': [],
            'annotation_type': [],
            'Feature Type': [],
            'PointsX1': [],
            'PointsY1': [],
            'PointsX2': [],
            'PointsY2': [],
            'PointsX3': [],
            'PointsY3': [],
            'PointsX4': [],
            'PointsY4': [],
            'Latitude': [],
            'Longitude': [],
            'Altitude': [],
            'confidence': []
        }
        self.feature_data_widt = {'Section_ID': [], 'RegionIndex': [], 'shoulderWidth': [], 'pavementWidth': []}
        self.frameN = 0
        self.script_dir = os.path.dirname(__file__)
        self.model_det = YOLO(
            os.path.normpath(os.path.join(self.script_dir, '..', '..', 'ai_model', '16mar_RF-wright.pt')))
        self.model_seg = YOLO(
            os.path.normpath(os.path.join(self.script_dir, '..', '..', 'ai_model', '16mar_segmentation.pt')))
        self.model_segm_soulder = YOLO(
            os.path.normpath(os.path.join(self.script_dir, '..', '..', 'ai_model', 'soulderWidt.pt')))

    def mask_frame(self, frame):
        """
        Masks a portion of the input frame by making a part of it black.        
        Args:
            frame (numpy array): The input video frame.
        Returns:
            numpy array: The masked frame.
        """
        height, width = frame.shape[:2]
        mask_frame = frame.copy()
        start_col = 0
        end_col = width - (width // 3)
        endrow = height // 3
        mask_frame[:, end_col:width] = [0, 0, 0]
        mask_frame[:endrow, :] = [0, 0, 0]
        mask_frame[1300:, :] = [0, 0, 0]
        return mask_frame

    def unmask_frame(self, frame, annotated_frame):
        """
        Unmasks the frame by restoring the original part of the frame that was masked.
        Args:
            frame (numpy array): The original frame.
            annotated_frame (numpy array): The annotated frame to be unmasked.
        Returns:
            numpy array: The unmasked frame.
        """
        height, width = frame.shape[:2]
        start_col = 0
        # end_col = width // 3
        end_col = width - (width // 3)
        endrow = height // 3
        annotated_frame[:, end_col:width] = frame[:, end_col:width]
        annotated_frame[:endrow, :] = frame[:endrow, :]
        annotated_frame[1300:, :] = frame[1300:, :]
        return annotated_frame

    def generate_new_csv_path(self):
        """
        Generates a new path for the output CSV by adding a 'soulder_' prefix to the file name.
        
        Returns:
            str: The new output CSV path with the updated file name.
        """
        path_parts = self.output_csv.rsplit('\\', 1)  # Split path into directory and file name
        file_name = path_parts[1]
        new_file_name = f"soulder_{file_name}"  # Add 'soulder_' prefix to the file name
        new_path = f"{path_parts[0]}\\{new_file_name}"  # Combine the directory with the new file name
        return new_path

    def calculate_width(self, frame, results_segm, frameN):
        """
        Calculates the pavement width from the segmentation results and annotates the frame with the width.
        Args:
            frame (numpy array): The input video frame.
            results_segm (list): Segmentation results.
            frameN (int): Frame number.
        Returns:
            numpy array: The frame annotated with the calculated width.
        """
        x1_diff_list_road = []
        y1_diff_list_road = []
        x2_diff_list_road = []
        y2_diff_list_road = []

        x1_diff_list_lane = []
        y1_diff_list_lane = []
        x2_diff_list_lane = []
        y2_diff_list_lane = []
        leftcordinates = []
        rightcordinates = []
        annotated_frame = frame.copy()
        for result_seg in results_segm:
            if result_seg.masks is not None:
                for mask, box in zip(result_seg.masks, result_seg.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0].cpu())
                    class_name = result_seg.names[class_id]

                    box_coords_xyxy = box.xyxy.cpu().numpy()
                    xs1, ys1, xs2, ys2 = int(box_coords_xyxy[0][0]), int(box_coords_xyxy[0][1]), int(
                        box_coords_xyxy[0][2]), int(box_coords_xyxy[0][3])

                    if class_id == 1:
                        x1_diff_list_road.append(xs1)
                        y1_diff_list_road.append(ys1)
                        x2_diff_list_road.append(xs2)
                        y2_diff_list_road.append(ys2)
                    elif class_id == 0:
                        x1_diff_list_lane.append(xs1)
                        y1_diff_list_lane.append(ys1)
                        x2_diff_list_lane.append(xs2)
                        y2_diff_list_lane.append(ys2)
        # # Sorting the coordinate lists
        # x1_diff_list_road.sort()
        # x1_diff_list_lane.sort()
        # x2_diff_list_road.sort()
        # x2_diff_list_lane.sort()

        try:
            x1_diff_list_lane, y1_diff_list_lane = self.sort_coordinate_lists(x1_diff_list_lane, y1_diff_list_lane)
            x2_diff_list_lane, y2_diff_list_lane = self.sort_coordinate_lists(x2_diff_list_lane, y2_diff_list_lane)
            x1_diff_list_road, y1_diff_list_road = self.sort_coordinate_lists(x1_diff_list_road, y1_diff_list_road)
            x2_diff_list_road, y2_diff_list_road = self.sort_coordinate_lists(x2_diff_list_road, y2_diff_list_road)

            x1_diff = abs(x1_diff_list_road[0] - x1_diff_list_lane[0])
            leftcordinates = (0, y2_diff_list_lane[0])
            rightcordinates = (x1_diff_list_lane[0], y2_diff_list_lane[0])
            x1_diff = Sheet.calcLength(leftcordinates, rightcordinates)
            pavementWidth = abs(x1_diff_list_lane[0] - x2_diff_list_lane[-1])
            leftcordinates = (x1_diff_list_lane[0], y2_diff_list_lane[0])
            rightcordinates = (x2_diff_list_lane[-1], y2_diff_list_lane[0])  # y2_diff_list_lane[-1]
            print(leftcordinates, rightcordinates)
            pavementWidth = Sheet.calcLength(leftcordinates, rightcordinates)
            # Annotating the frame with distance information
            cv2.putText(annotated_frame, f"Soulder Distance is {x1_diff}", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 4)
            cv2.line(annotated_frame, (x1_diff_list_road[0], 2000), (x1_diff_list_lane[0], 2000), (255, 0, 255), 2)
            cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 4)
            # cv2.line(annotated_frame, (x1_diff_list_lane[0], 1800), (x2_diff_list_lane[-1], 1800), (100, 200, 100), 4)
            cv2.line(annotated_frame, leftcordinates, rightcordinates, (100, 200, 100), 4)
        except Exception as e:
            print(e)
            x1_diff = [0]
            pavementWidth = [0]
            cv2.putText(annotated_frame, f"soulder Distance is {x1_diff}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 4)
            cv2.putText(annotated_frame, f"Pavement Distance is {pavementWidth}", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 4)
        self.feature_data_widt['Section_ID'].append(self.sectionID)
        self.feature_data_widt['RegionIndex'].append(frameN)
        self.feature_data_widt['shoulderWidth'].append(0)  # x1_diff
        self.feature_data_widt['pavementWidth'].append(pavementWidth[0])
        return annotated_frame

    def sort_coordinate_lists(self, x_list, y_list):
        """
        Sorts the x and y coordinate lists based on the x values.
        
        Args:
            x_list (list): List of x coordinates.
            y_list (list): List of y coordinates.
        
        Returns:
            tuple: Sorted x and y lists.
        """
        combined = list(zip(x_list, y_list))
        combined = sorted(combined, key=lambda pair: pair[0])
        x_list, y_list = zip(*combined)
        return list(x_list), list(y_list)

    def process_frame(self, frame, frameN):
        """
        Processes a single frame for width calculation.
        Args:
            frame (numpy array): The video frame to be processed.
            frameN (int): The frame number.
        Returns:
            Annotated frame.
        """
        mask_frame = self.mask_frame(frame)
        results_segm_soulder = self.model_segm_soulder(mask_frame, conf=0.4, iou=0.47, imgsz=640, device=0,
                                                       show_boxes=True)  ####new added
        annotated_frame = results_segm_soulder[0].plot(boxes=True)
        annotated_frame1 = self.unmask_frame(frame, annotated_frame)
        annotated_frame1 = self.calculate_width(annotated_frame1, results_segm_soulder, frameN)  ####new
        return annotated_frame1

    def save_features_to_csv(self):
        """ Saves the extracted features to a CSV file."""
        dataFrame = pd.DataFrame(self.feature_data)
        dataFrame.to_csv(self.output_csv, index=False)
        print("Data saved to CSV:", self.output_csv)
        dataFrame = pd.DataFrame(self.feature_data_widt)
        dataFrame.to_csv(self.output_csv_, index=False)
        print("Data saved to CSV:", self.output_csv_)

    def process_video(self):
        image_dir = self.video_path
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))  # Adjust the extension if needed
        self.image_paths = sorted([p for p in glob(os.path.join(image_dir, "*.jpg")) if "RL" in p])
        i = 0
        # cv2.namedWindow("Road Furniture Detection", cv2.WINDOW_NORMAL)
        for image_path in self.image_paths:
            # image_path = "G:/NEHRU RING ROAD_2024-09-04_15-29-02/SECTION-1/pcams/pcam-0000124-RL.jpg"
            frame = cv2.imread(image_path)
            print(f'Invertory process:{i}', flush=True)
            i += 1
            startTime = time.time()
            annotated_frame1 = self.process_frame(frame, self.frameN)
            # self.extract_features(results_segm, results_det, frame, self.frameN)
            endTime = time.time()
            frameProcessTime = endTime - startTime
            print(f"Processed {image_path} in {frameProcessTime:.4f} seconds")
            self.total_processing_time += frameProcessTime
            self.frame_processing_timeList.append(frameProcessTime)
            self.frameN += 1
            # cv2.imshow("Road Furniture Detection", annotated_frame1)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # cap.release()
        cv2.destroyAllWindows()
        self.save_features_to_csv()


if __name__ == '__main__':
    video_path = "G:/NEHRU RING ROAD_2024-09-04_15-29-02/SECTION-1/pcams"  # Update this path to your image directory
    gps_data_path = 'G:/NEHRU RING ROAD_2024-09-04_15-29-02/SECTION-1/interpolated_track.pkl'
    output_csv = 'G:\\NEHRU RING ROAD_2024-09-04_15-29-02\\SECTION-1\\80be-4c33-a4c5-c826176986ae.csv'
    gps_data = pd.read_pickle(gps_data_path)
    sectionID = '33452aee-80be-4c33-a4c5-c826176986ae'
    inventory_features = InventoryFeatures_Outer(video_path, gps_data, output_csv, sectionID)
    inventory_features.process_video()
