from yolov5_src import YOLOV5Impl
from grpcs.task_coordinate.task_coordinate_server import task_coordinate_serve
from utils.task_ctrl import TaskCtrl

def connect_consul():
    from common.consul_client import ConsulClient
    from common.utils import get_local_ip_address
    host = get_local_ip_address()
    port = 5000
    service_name = 'target detection'
    consul_client = ConsulClient()
    consul_client\
        .set_service_address(host)\
        .set_service_port(port)\
        .set_service_id(f'{service_name}-{host}-{port}')\
        .set_service_name(service_name)\
        .set_service_tags(['yolov5', 'grpc'])\
        .register_service()

if __name__ == '__main__':
    connect_consul()
    
    yolov5_builder = YOLOV5Impl.YOLOV5Builder()
    yolov5_builder._device = 'cuda:0'
    yolov5_impl = yolov5_builder.build()
    yolov5_impl.load_model()

    task_ctrl = TaskCtrl()
    task_ctrl.set_yolov5_impl(yolov5_impl)
    task_ctrl.listening()
    task_coordinate_serve('0.0.0.0', '5000', 10)
