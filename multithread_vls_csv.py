import cv2
import csv
import concurrent.futures
import os
from ultralytics import YOLO

model_cls = YOLO(r'cls/asp_conc3/weights/best.pt')


section_uuid_map = {
    'SECTION-1': 'e515e4a3-c173-4e31-821f-3640032d1a0b',
    'SECTION-2': 'd63abd4e-a5dd-4ba7-8ac6-6108e7089229',
    'SECTION-3': '25239e5b-4bd4-426e-a2cb-f441aebc2fae',
    'SECTION-4': '651a3ed3-d9a3-44df-8d57-79f397723680',
    'SECTION-5': '214ec2c1-1bb4-4b7c-a05f-f2a8eae5bba9',
    'SECTION-6': 'c5637f92-6fd5-410a-883e-02a076b9ba79',
}

def process_video_section(section_id, video_path):
    # Get the UUID for the current section
    uuid = section_uuid_map.get(section_id)
    
    # Define the output directory on the D drive
    output_dir = f'D:/KISAN PATH-OUTER RING ROAD_2024-10-02_08-35-28/{section_id}/reportNew/csv_reports'
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    csv_file_path = os.path.join(output_dir, 'road_type.csv')
    
    with open(csv_file_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header with UUID
        csv_writer.writerow(['Section_ID', 'Region_index', 'Road type'])
        
        cap = cv2.VideoCapture(video_path)  # Read video from E drive
        frame_number = 0 

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                if frame is not None and frame.shape[1] == 400 and frame.shape[0] == 1000:
                    frame = cv2.resize(frame, (1280, 720))
                results = model_cls.predict(frame, conf=0.3, iou=0.4, imgsz=320, device=0)
                for r in results:
                    top1_prob = r.probs.top1
                    road_type = 'Asphalt Road' if top1_prob == 0 else 'Concrete Road'
                    csv_writer.writerow([uuid, frame_number, road_type])
                frame_number += 1  # Increment frame counter
            else:
                break

        cap.release()    

def main():
    video_paths = {
        'SECTION-1': r'Z:\SA_DATA_2024\KISAN PATH-OUTER RING ROAD_2024-10-02_08-35-28/SECTION-1/KISAN PATH-OUTER RING ROAD_SURFACE_SECTION-1.mp4',
        'SECTION-2': r'Z:\SA_DATA_2024\KISAN PATH-OUTER RING ROAD_2024-10-02_08-35-28/SECTION-2/KISAN PATH-OUTER RING ROAD_SURFACE_SECTION-2.mp4',
        'SECTION-3': r'Z:\SA_DATA_2024\KISAN PATH-OUTER RING ROAD_2024-10-02_08-35-28/SECTION-3/KISAN PATH-OUTER RING ROAD_SURFACE_SECTION-3.mp4',
        'SECTION-4': r'Z:\SA_DATA_2024\KISAN PATH-OUTER RING ROAD_2024-10-02_08-35-28/SECTION-4/KISAN PATH-OUTER RING ROAD_SURFACE_SECTION-4.mp4',
        'SECTION-5': r'Z:\SA_DATA_2024\KISAN PATH-OUTER RING ROAD_2024-10-02_08-35-28/SECTION-5/KISAN PATH-OUTER RING ROAD_SURFACE_SECTION-5.mp4',
        'SECTION-6': r'Z:\SA_DATA_2024\KISAN PATH-OUTER RING ROAD_2024-10-02_08-35-28/SECTION-6/KISAN PATH-OUTER RING ROAD_SURFACE_SECTION-6.mp4',
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for section_id, video_path in video_paths.items():
            futures.append(executor.submit(process_video_section, section_id, video_path))

        # Wait for all threads to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise exceptions if any occurred in the threads
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
