from ultralytics import YOLO
# from ultralytics import settings
# View all settings
# print(settings)
# settings.reset()
model = YOLO('yolov8m-seg.pt')

if __name__ == '__main__':
    train_model = model.train(data=r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/data/roadfurniture_median.yaml',
                              cfg = r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/cfg/shoulderWidth.yaml',
                              project = 'shoulderWidth' , name = '24july' , imgsz = 640,
                              device = 0, epochs=500)



