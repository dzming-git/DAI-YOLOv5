from concurrent import futures
import time
import grpc
import traceback
from generated.protos.service_coordinator import service_coordinator_pb2, service_coordinator_pb2_grpc
from typing import Dict
from src.task_manager.task_manager import TaskManager, TaskInfo
from src.model_manager.model_manager import ModelManager
from src.model_manager import model_manager as mm

VALID_PRE_SERVICE = ['image harmony']

class ServiceCoordinatorServer(service_coordinator_pb2_grpc.CommunicateServicer):
    def informPreviousServiceInfo(self, request, context):
        print(request)
        response_code = 200
        response_message = ''
        
        try:
            task_manager = TaskManager()
            task_id = request.taskId
            task_info: TaskInfo = None

            if task_id in task_manager.tasks:
                # task_id在task_manager.tasks存在
                # 修改已有任务的参数
                task_info = task_manager.tasks[task_id]
            else:
                # 新的task_id
                task_info = TaskInfo()
                task_manager.incomplete_tasks[task_id] = task_info
                task_manager.tasks[task_id] = task_info
                task_info.id = request.taskId
            assert request.preServiceName in VALID_PRE_SERVICE, 'invalid pre service\n'
            
            task_info.set_pre_service(
                pre_service_name=request.preServiceName,
                pre_service_ip=request.preServiceIp,
                pre_service_port=request.preServicePort)
            if task_id in task_manager.incomplete_tasks:
                ok, msg = task_info.check()
                if ok:
                    task_manager.tasks_queue.put(task_info)
                    task_manager.incomplete_tasks.pop(task_id)

        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()

        response = service_coordinator_pb2.InformPreviousServiceInfoResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def informCurrentServiceInfo(self, request, context):
        print(request)
        response_code = 200
        response_message = ''
        # TODO: 目前只支持weight、ConnectID
        try:
            task_manager = TaskManager()
            task_id = request.taskId
            task_info: TaskInfo = None

            if task_id in task_manager.tasks:
                # task_id在task_manager.tasks存在
                # 修改已有任务的参数
                task_info = task_manager.tasks[task_id]
            else:
                # 新的task id
                task_info = TaskInfo()
                task_manager.incomplete_tasks[task_id] = task_info
                task_manager.tasks[task_id] = task_info
                task_info.id = request.taskId
            
            if task_id in task_manager.incomplete_tasks:
                # task_id在task_manager.incomplete_tasks存在
                # 修改已有任务的参数
                task_info = task_manager.incomplete_tasks[task_id]

            weight = ''
            loader_args_hash = 0
            device = 'cpu'  # 默认cpu
            for arg in request.args:
                if 'Weight' == arg.key:
                    weight = arg.value
                if 'LoaderArgsHash' == arg.key:
                    loader_args_hash = int(arg.value)
                if 'Device' == arg.key:
                    device = arg.value
            # assert weight, 'Error: Missing parameter \'Weight\'\n'
            # assert connect_id, 'Error: Missing parameter \'ConnectID\'\n'
            task_info.set_cur_service(
                weight=weight,
                device=device,
                loader_args_hash=loader_args_hash
            )
            if task_id in task_manager.incomplete_tasks:
                ok, msg = task_info.check()
                if ok:
                    task_manager.tasks_queue.put(task_info)
                    task_manager.incomplete_tasks.pop(task_id)
        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()

        response = service_coordinator_pb2.InformPreviousServiceInfoResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response

    def start(self, request, context):
        response_code = 200
        response_message = ''
        try:
            task_manager = TaskManager()
            model_manager = ModelManager()
            task_id = request.taskId
            assert task_id in task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            detector = task_manager.tasks[task_id].detector
            weight = detector._weight
            device = detector._device
            dnn = detector._dnn
            half = detector._half
            model_info = model_manager.get_model_info(weight, device, dnn, half)
            model_state = model_info.get_state()
            if model_state == mm.NOT_LOADED:
                detector.load_model()
            elif model_state == mm.LOADING:
                response_message += 'Model is loading.\n'
            elif model_state == mm.LOADING_COMPLETED:
                response_message += 'Model loading completed.\n'
            # TODO 简单实现，阻塞等待
            while 1:
                model_state = model_info.get_state()
                if model_state == mm.LOADING_COMPLETED:
                    break
                time.sleep(1)
        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()
        response = service_coordinator_pb2.StartResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def join_in_server(self, server):
        service_coordinator_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
