from yolov5_src import YOLOV5Impl
from src.grpc.service_coordinate.service_coordinate_server import service_coordinate_serve
from src.task_ctrl.task_ctrl import TaskCtrl
from src.config.config import Config

config = Config()

def connect_consul():
    from src.consul import ConsulClient
    from src.consul import ServiceInfo
    from src.utils import get_local_ip_address

    consul_client = ConsulClient()
    consul_client.consul_ip = config.consul_ip
    consul_client.consul_port = config.consul_port
    host = get_local_ip_address()
    service_name = config.service_name
    service_port = config.service_port
    service_info = ServiceInfo()
    service_info.service_ip = host
    service_info.service_port = service_port
    service_info.service_name = service_name

    for weight in config.weights:
        service_info.service_id = f'{service_name}-{weight.file}-{host}-{service_port}'
        service_info.service_tags = config.service_tags
        for label in weight.labels:
            service_info.service_tags.append(f'label:{label}')

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
    service_coordinate_serve('0.0.0.0', '5000', 10)
