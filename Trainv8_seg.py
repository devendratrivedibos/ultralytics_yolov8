from ultralytics import YOLO
# from ultralytics import settings
# View all settings
# print(settings)
# settings.reset()
model = YOLO('yolov8m-seg.pt')

if __name__ == '__main__':
    train_model = model.train(data=r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/data/roadfurniture_seg.yaml',
                              cfg = r'E:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/cfg/default24feb_seg.yaml',
                              project = 'segmentation' , name = '29apr' , imgsz = 640,
                              device = 0, batch = 8,epochs=500)

