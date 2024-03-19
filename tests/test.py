import cv2
from src.wrapper.yolov5_detector import YOLOv5Detector

if __name__ == '__main__':
    detector_builder = YOLOv5Detector.YOLOv5Builder()
    detector_builder.device_str = 'cuda:0'
    detector_builder.save_image_to_cache = True
    detector = detector_builder.build()
    
    image = cv2.imread('yolov5/data/images/bus.jpg')
    image_id = 1

    detector.load_model()
    detector.add_image_id(image_id)
    detector.add_image(image_id, image)
    detector.detect_by_image_id(image_id)
    result = detector.get_result_by_image_id(image_id)
    print(result)
    
    labeled_image = detector.get_labeled_image_by_image_id(image_id)
    cv2.imshow('labeled_image', labeled_image)
    cv2.waitKey(0)
    