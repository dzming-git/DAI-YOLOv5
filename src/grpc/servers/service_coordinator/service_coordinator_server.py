from generated.protos.service_coordinator import service_coordinator_pb2, service_coordinator_pb2_grpc
from src.task_manager.task_manager import TaskManager, TaskInfo

VALID_PRE_SERVICE = ['image harmony gRPC']

class ServiceCoordinatorServer(service_coordinator_pb2_grpc.CommunicateServicer):
    def informPreviousServiceInfo(self, request, context):
        print(request)
        response_code = 200
        response_message = ''
        response = service_coordinator_pb2.InformPreviousServiceInfoResponse()
        
        try:
            task_manager = TaskManager()
            task_id = request.taskId

            if task_id in task_manager.tasks:
                raise ValueError(f'Task ID {task_id} is already initialized.')
            task_manager.tasks[task_id] = TaskInfo(request.taskId)

            # 检查preServiceName是否在VALID_PRE_SERVICE列表中
            if request.preServiceName not in VALID_PRE_SERVICE:
                raise ValueError(f'Invalid pre service: {request.preServiceName}')

            args = {}
            for arg in request.args:
                args[arg.key] = arg.value
            task_manager.tasks[task_id].set_pre_service(
                pre_service_name=request.preServiceName,
                pre_service_ip=request.preServiceIp,
                pre_service_port=request.preServicePort,
                args=args)

        except Exception as e:
            response.response.code = 400
            response.response.message = str(e)
            return response

        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def informCurrentServiceInfo(self, request, context):
        print(request)
        response_code = 200
        response_message = ''
        response = service_coordinator_pb2.InformPreviousServiceInfoResponse()
        try:
            task_manager = TaskManager()
            task_id = request.taskId

            if task_id not in task_manager.tasks:
                task_manager.tasks[task_id] = TaskInfo(request.taskId)
            args = {}
            for arg in request.args:
                args[arg.key] = arg.value
            task_manager.tasks[task_id].set_cur_service(args)
        except Exception as e:
            response.response.code = 400
            response.response.message = str(e)
            return response

        response.response.code = response_code
        response.response.message = response_message
        return response

    def start(self, request, context):
        response_code = 200
        response_message = ''
        response = service_coordinator_pb2.StartResponse()
        try:
            task_manager = TaskManager()
            task_id = request.taskId
            if task_id not in task_manager.tasks:
                raise Exception('ERROR: The task ID does not exist.')
            task_manager.tasks[task_id].start()
        except Exception as e:
            response.response.code = 400
            response.response.message = str(e)
            return response
        
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def stop(self, request, context):
        response_code = 200
        response_message = ''
        response = service_coordinator_pb2.StartResponse()
        task_id = request.taskId
        try:
            task_manager = TaskManager()
            task_manager.stop_task(task_id)
        except Exception as e:
            response.response.code = 400
            response.response.message = str(e)
            return response
        
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def join_in_server(self, server):
        service_coordinator_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
