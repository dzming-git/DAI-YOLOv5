from concurrent import futures
import time
import grpc
from generated.protos.service_coordinator import service_coordinator_pb2, service_coordinator_pb2_grpc
from typing import Dict
from src.task_manager.task_manager import TaskManager, TaskInfo

VALID_PRE_SERVICE = ['image harmony']

class ServiceCoordinatorServer(service_coordinator_pb2_grpc.CommunicateServicer):
    def informPreviousServiceInfo(self, request, context):
        response_code = 200
        response_message = ''
        
        try:
            task_manager = TaskManager()
            task_id = request.taskId
            assert task_id not in task_manager.tasks, 'task id already exists\n'
            if task_id not in task_manager.incomplete_tasks:
                task_manager.incomplete_tasks[task_id] = TaskInfo()
                task_manager.incomplete_tasks[task_id].id = request.taskId
            assert request.preServiceName in VALID_PRE_SERVICE, 'invalid pre service\n'
            
            task_manager.incomplete_tasks[task_id].set_pre_service(
                pre_service_name=request.preServiceName,
                pre_service_ip=request.preServiceIp,
                pre_service_port=request.preServicePort)
            if (0 == task_manager.incomplete_tasks[task_id].step):
                task_info = task_manager.incomplete_tasks[task_id]
                task_manager.tasks[task_id] = task_info
                task_manager.tasks_queue.put(task_info)
                task_manager.incomplete_tasks.pop(task_id)
        except Exception as e:
            response_code = 400
            response_message += e

        response = service_coordinator_pb2.InformPreviousServiceInfoResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def informCurrentServiceInfo(self, request, context):
        response_code = 200
        response_message = ''
        # TODO: 目前只支持weight、ConnectID
        try:
            task_manager = TaskManager()
            task_id = request.taskId
            # assert task_id not in task_manager.tasks, 'task id already exists\n'
            if task_id not in task_manager.incomplete_tasks:
                task_manager.incomplete_tasks[task_id] = TaskInfo()
                task_manager.incomplete_tasks[task_id].id = request.taskId
            weight = ''
            connect_id = 0
            for arg in request.args:
                if 'Weight' == arg.key:
                    weight = arg.value
                if 'ConnectID' == arg.key:
                    connect_id = int(arg.value)
            assert weight, 'Error: Missing parameter \'Weight\'\n'
            assert connect_id, 'Error: Missing parameter \'ConnectID\'\n'
            task_manager.incomplete_tasks[task_id].set_cur_service(
                weight=weight,
                connect_id=connect_id
            )
            if (0 == task_manager.incomplete_tasks[task_id].step):
                task_info = task_manager.incomplete_tasks[task_id]
                task_manager.tasks[task_id] = task_info
                task_manager.tasks_queue.put(task_info)
                task_manager.incomplete_tasks.popitem(task_id)
        except Exception as e:
            response_code = 400
            response_message += e

        response = service_coordinator_pb2.InformPreviousServiceInfoResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def join_in_server(self, server):
        service_coordinator_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
