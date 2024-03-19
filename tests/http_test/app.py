from src.wrapper.yolov5_detector import YOLOv5Detector
from src.http_server.service import YOLOv5Service

if __name__ == '__main__':
    detector_builder = YOLOv5Detector.YOLOv5Builder()
    detector_builder.device_str = 'cuda:0'
    detector_builder.save_image_to_cache = True
    detector = detector_builder.build()
    
    service_builder = YOLOv5Service.SingletonBuilder()
    service_builder.host = '0.0.0.0'
    service_builder.template_folder = '/workspace/tests/http_test/templates'
    service_builder.build()
    service = service_builder.get_instance()
    service.set_detector(detector)
    service.start_server()
