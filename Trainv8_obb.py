from ultralytics import YOLO


# model = YOLO(r'yolov8m-obb.pt')
model = YOLO(r'w:/Devendra_Files/ultralytics-main/ultralytics-main/24Feb_obb/train2/weights/last.pt')

if __name__ == '__main__':
    # #train_model = model.train(data=r'W:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/data/roadfurniture_obb.yaml' ,
    #                           cfg = r'W:/Devendra_Files/ultralytics-main/ultralytics-main/ultralytics/cfg/default24feb_obb.yaml',
    #                            device = 0, imgsz=640, epochs = 500)
    

    # model = YOLO("last.pt")
    model.train(resume=True)



