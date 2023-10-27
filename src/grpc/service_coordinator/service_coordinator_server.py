from concurrent import futures
import time
import grpc
from generated.protos.service_coordinator import service_coordinator_pb2, service_coordinator_pb2_grpc
from typing import Dict
from src.task_ctrl.task_ctrl import TaskCtrl, TaskInfo

VALID_PRE_SERVICE = ['image harmony']

class ServiceCoordinatorServer(service_coordinator_pb2_grpc.CommunicateServicer):
    def informPreviousServiceInfo(self, request, context):
        response_code = 200
        response_message = ''
        
        try:
            task_info = TaskInfo()
            task_info.id = request.taskId
            task_info.pre_service_name = request.preServiceName
            assert request.preServiceName in VALID_PRE_SERVICE, 'invalid pre service\n'
            task_info.pre_service_ip = request.preServiceIp
            task_info.pre_service_port = request.preServicePort
            for arg in request.args:
                # 特殊参数
                if 'ImgType' == arg.key:
                    task_info.img_type = arg.value
                    continue
                # 普通参数
                task_info.img_args[arg.key] = arg.value
            assert task_info.img_type, 'ImgType is not set\n'
            task_ctrl = TaskCtrl()
            task_ctrl.tasks_queue.put(task_info)
        except Exception as e:
            response_code = 400
            response_message += e

        response = service_coordinator_pb2.InformPreviousServiceInfoResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response

def service_coordinator_serve(ip, port, maxWorkers):
    address = f'{ip}:{port}'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=maxWorkers))
    service_coordinator_pb2_grpc.add_CommunicateServicer_to_server(ServiceCoordinatorServer(), server)
    server.add_insecure_port(address)
    server.start()

    print(f'service_coordinator_server listening on {address}')
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
