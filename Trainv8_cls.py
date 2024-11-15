""" train classification"""
from ultralytics import YOLO
import torch
model = YOLO('yolov8s-cls.pt')
torch.cuda.empty_cache()


if __name__ == '__main__':
    train_model = model.train(data=r'D:/DATA_RF/Asphat_concrete/', project='cls', name='asp_conc',
                              imgsz=320, batch=32, epochs=500, device=0)
