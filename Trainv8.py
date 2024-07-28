from ultralytics import YOLO
from ultralytics import settings

# View all settings
print(settings)
model = YOLO('yolov8m.pt')

#train_model = model.train(data=r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/data/9Feb_roadfurniture.yaml')
if __name__ == '__main__':
    train_model = model.train(data=r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/data/9Feb_roadfurniture.yaml')



