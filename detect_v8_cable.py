from ultralytics import YOLO

# model_seg = YOLO('segmentation/pot_patch7/weights/best.pt')           
model_seg = YOLO(r'D:\Devendra_Files\ultralytics_yolov8\segmentation\lanemark2\weights/best.pt')

if __name__ == '__main__':

    source = r"Z:\SA_DATA_2024\LUCKNOW-RAEBARELI_2024-10-02_12-14-16\LUCKNOW-RAEBARELI_SURFACE_SECTION-8.mp4"
    results = model_seg(source, imgsz=640, conf=.3, device=0, save=True, show=True)  # generator of Results objects
    for r in results:
        next(results)
