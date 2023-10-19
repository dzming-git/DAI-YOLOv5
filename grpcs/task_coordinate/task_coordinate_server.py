from concurrent import futures
import time
import grpc
from common.grpcs.task_coordinate import task_coordinate_pb2, task_coordinate_pb2_grpc
from typing import Dict
from utils.task_ctrl import TaskCtrl, TaskInfo

class TaskCoordinateServer(task_coordinate_pb2_grpc.CommunicateServicer):
    def informPreviousServiceInfo(self, request, context):
        response = task_coordinate_pb2.InformPreviousServiceInfoResponse()
        response.response.code = 200
        task_info = TaskInfo()
        task_info.id = request.taskId
        task_info.pre_service_ip = request.preServiceIp
        task_info.pre_service_port = request.preServicePort
        for arg in request.args:
            # 特殊参数
            if 'ImgType' == arg.key:
                task_info.img_type = arg.value
                continue
            # 普通参数
            task_info.img_args[arg.key] = arg.value
        task_ctrl = TaskCtrl()
        task_ctrl.tasks_queue.put(task_info)
        return response

def task_coordinate_serve(ip, port, maxWorkers):
    address = f'{ip}:{port}'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=maxWorkers))
    task_coordinate_pb2_grpc.add_CommunicateServicer_to_server(TaskCoordinateServer(), server)
    server.add_insecure_port(address)
    server.start()

    print(f'task_coordinate_server listening on {address}')
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
