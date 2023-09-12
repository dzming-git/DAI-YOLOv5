from yolov5_src import YOLOV5Impl
from service import YOLOv5Service

def connect_consul():
    pass

if __name__ == '__main__':
    yolov5_builder = YOLOV5Impl.YOLOV5Builder()
    yolov5_builder._device = 'cuda:0'
    yolov5_impl = yolov5_builder.build()
    


    yolov5_service_builder = YOLOv5Service.SingletonBuilder()
    yolov5_service_builder.build()
    yolov5_service = yolov5_service_builder.get_instance()
    yolov5_service.set_yolov5_impl(yolov5_impl)
    yolov5_service.start_server()
