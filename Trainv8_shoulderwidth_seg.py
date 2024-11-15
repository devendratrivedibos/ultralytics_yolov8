from ultralytics import YOLO

model = YOLO('yolov8s-seg.pt')

if __name__ == '__main__':
    train_model = model.train(data=r'C:/Users/Admin/Downloads/Median_Width.v2i.yolov11/data.yaml',
                              # cfg = r'D:/Devendra_Files/ultralytics_yolov8/ultralytics/cfg/shoulderWidth.yaml',
                              project = 'medianWidth' , name = '31oct' , imgsz = 640,
                              device = 0, epochs=500)



