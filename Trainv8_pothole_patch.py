from ultralytics import YOLO
from ultralytics import settings
# View all settings
print(settings)
settings.reset()
model = YOLO('yolo11s-seg.yaml')
if __name__ == '__main__':
    train_model = model.train(data=r'D:/Trial_Potholr_patch/YOLODataset/dataset.yaml',
                              project = 'segmentation' , name = 'pot_patch' , imgsz = 640, batch = 32,
                              device = 0, epochs=500, flipud=0.2 , fliplr = 0.2)
    # train_model = model.train(resume=True)



