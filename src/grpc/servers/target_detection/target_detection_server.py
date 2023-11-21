from generated.protos.target_detection import target_detection_pb2, target_detection_pb2_grpc
from typing import Dict
from src.task_manager.task_manager import TaskManager
from src.config.config import Config

class TargetDetectionServer(target_detection_pb2_grpc.CommunicateServicer):
    def getResultMappingTable(self, request, context):
        response_code = 200
        response_message = ''
        labels = []
        try:
            task_manager = TaskManager()
            config = Config()
            task_id = request.taskId
            assert task_id in task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            weight = task_manager.tasks[task_id].weight
            assert weight in config.weights_map, f'ERROR: The configuration file does not contain the weight: {weight}.\n'
            weight_info = config.weights_map[weight]
            labels = weight_info.labels
            
        except Exception as e:
            response_code = 400
            response_message += e

        response = target_detection_pb2.GetResultMappingTableResponse()
        response.response.code = response_code
        response.response.message = response_message
        for label in labels:
            response.labels.append(label)
        return response
    
    def joinInServer(self, server):
        target_detection_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
