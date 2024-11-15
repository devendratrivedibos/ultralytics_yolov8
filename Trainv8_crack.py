from ultralytics import YOLO
from ultralytics import settings
# View all settings
print(settings)
settings.reset()
model = YOLO('yolov8s-seg.pt')
# model = YOLO('yolov8s-seg.yaml')

if __name__ == '__main__':
    train_model = model.train(data=r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/data/distress_dataset.yaml',
                              cfg=r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/cfg/distress_dataset.yaml',
                              project='cracks', imgsz=640, batch=8,
                              device=0, epochs=500, flipud=0.3, fliplr=0.2)


