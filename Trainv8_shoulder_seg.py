from ultralytics import YOLO
# from ultralytics import settings
# View all settings
# print(settings)
# settings.reset()
model = YOLO('yolov8m-seg.pt')

if __name__ == '__main__':
    train_model = model.train(data=r'D:/Devendra_Files/ultralytics_yolov8/ultralytics/data/shoulderWidth.yaml',
                              cfg = r'D:/Devendra_Files/ultralytics_yolov8/ultralytics/cfg/shoulderWidth.yaml',
                              project = 'shoulderWidth', name='7nov', imgsz = 640,
                              device = 0, epochs=500)



