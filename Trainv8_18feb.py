from ultralytics import YOLO
# model = YOLO(r'yolov8m.pt')

model = YOLO(r'E:/Devendra_Files/ultralytics-main/ultralytics-main/roadFurniture/16mar/weights/best.pt')
# model = YOLO(r'yolov8l.pt')


if __name__ == '__main__':
    train_model = model.train(data=r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/data/14Feb_roadfurniture.yaml' ,
                              cfg = r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/cfg/default22feb.yaml',
                              project = 'roadFurniture', name = '22mar', batch = 8,
                              device = 0)



