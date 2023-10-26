from yolov5_src import YOLOV5Impl
from grpcs.task_coordinate.task_coordinate_server import task_coordinate_serve
from utils.task_ctrl import TaskCtrl
from config import Config

config = Config()

def connect_consul():
    from common.consul_client import ConsulClient
    from common.service_info import ServiceInfo
    from common.utils import get_local_ip_address

    consul_client = ConsulClient()
    consul_client.consul_ip = config.consul_ip
    consul_client.consul_port = config.consul_port

    service_info = ServiceInfo()
    host = get_local_ip_address()
    service_name = config.service_name
    service_port = config.service_port
    service_info.service_ip = host
    service_info.service_port = service_port
    service_info.service_name = service_name
    service_info.service_id = f'{service_name}-{host}-{service_port}'
    service_info.service_tags = config.service_tags

    consul_client.register_service(service_info)

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
