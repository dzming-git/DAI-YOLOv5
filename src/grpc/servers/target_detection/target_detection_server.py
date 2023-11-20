from generated.protos.target_detection import target_detection_pb2, target_detection_pb2_grpc
from typing import Dict
from task_manager.task_manager import TaskCtrl, TaskInfo

VALID_PRE_SERVICE = ['image harmony']

class TargetDetectionServer(target_detection_pb2_grpc.CommunicateServicer):

    
    def joinInServer(self, server):
        target_detection_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
