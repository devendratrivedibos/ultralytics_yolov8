from ultralytics import YOLO

model = YOLO('yolov8s-seg.pt')

if __name__ == '__main__':
    train_model = model.train(
        data=r'D:\Devendra_Files\ultralytics_yolov8/ultralytics/data/lanemark_seg.yaml',
        project='segmentation', name='lanemark', imgsz=640,
        device=0, batch=8, epochs=500)
