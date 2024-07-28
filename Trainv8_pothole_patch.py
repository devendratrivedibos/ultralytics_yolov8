from ultralytics import YOLO
from ultralytics import settings
# View all settings
print(settings)
settings.reset()
# model = YOLO('yolov8m-seg.pt')
model = YOLO('segmentation/pot_patch/weights/best.pt') 
if __name__ == '__main__':
    train_model = model.train(data=r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/data/distress_dataset.yaml',
                              cfg = r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/cfg/distress_dataset.yaml',
                              project = 'segmentation' , name = 'pot_patch' , imgsz = 1024, batch = 4,
                              device = 0, epochs=500, flipud=0.5 , fliplr = 0.2)
    # train_model = model.train(resume=True)



