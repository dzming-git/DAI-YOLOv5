import cv2
# import sys
# import os

# project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5_src'))
# sys.path.insert(0, project_path)
from src.wrapper.yolov5_detector import YOLOv5Detector

builder = YOLOv5Detector.YOLOv5Builder()
builder.device = 'cuda:0'
builder.weight = 'yolov5s.pt'
detector = builder.build()

image = cv2.imread('yolov5_src/data/images/bus.jpg')

detector.load_model()
detector.add_uid(1)
detector.add_img(1, image)
detector.detect_by_uid(1)
result = detector.get_result_by_uid(1)
print(result)
