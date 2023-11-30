from generated.protos.target_detection import target_detection_pb2, target_detection_pb2_grpc
from typing import Dict
import time
from src.task_manager.task_manager import TaskManager
from src.detector_manager.detector_manager import DetectorManager
from src.config.config import Config
from src.wrapper import yolov5_detector as y5d

class TargetDetectionServer(target_detection_pb2_grpc.CommunicateServicer):
    def __init__(self):
        self.task_manager = TaskManager()
        self.detector_manager = DetectorManager()
        self.config = Config()
        
    def getResultMappingTable(self, request, context):
        response_code = 200
        response_message = ''
        labels = []
        try:
            task_id = request.taskId
            assert task_id in self.task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            weight = self.task_manager.tasks[task_id].weight
            assert weight in self.config.weights_map, f'ERROR: The configuration file does not contain the weight: {weight}.\n'
            weight_info = self.config.weights_map[weight]
            labels = weight_info.labels
        except Exception as e:
            response_code = 400
            response_message += str(e)

        response = target_detection_pb2.GetResultMappingTableResponse()
        response.response.code = response_code
        response.response.message = response_message
        for label in labels:
            response.labels.append(label)
        return response
    
    def checkModelState(self, request, context):
        response_code = 200
        response_message = ''
        modelState = target_detection_pb2.ModelState.NotSet
        results = []
        try:
            task_id = request.taskId
            assert task_id in self.task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            weight = self.task_manager.tasks[task_id].weight
            if weight not in self.detector_manager.detector_info_map:
                modelState = target_detection_pb2.ModelState.NotSet
            else:
                detector_info = self.detector_manager.detector_info_map[weight]
                detector = detector_info.detector
                model_state = detector.check_model_state()
                if model_state == y5d.NOT_LOADED:
                    modelState = target_detection_pb2.ModelState.NotLoaded
                elif model_state == y5d.LOADING:
                    modelState = target_detection_pb2.ModelState.Loading
                elif model_state == y5d.LOADING_COMPLETED:
                    modelState = target_detection_pb2.ModelState.LoadingCompleted
        except Exception as e:
            response_code = 400
            response_message += str(e)

        response = target_detection_pb2.CheckModelStateResponse()
        response.response.code = response_code
        response.response.message = response_message
        response.modelState = modelState
        return response

    def getResultIndexByImageId(self, request, context):
        response_code = 200
        response_message = ''
        results = []
        try:
            task_id = request.taskId
            image_id = request.imageId
            wait = request.wait
            assert task_id in self.task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            weight = self.task_manager.tasks[task_id].weight
            assert weight in self.detector_manager.detector_info_map, f'ERROR: can not find {weight} in detector_info_map.\n'
            detector = self.detector_manager.detector_info_map[weight].detector
            image_id_exist = detector.check_uid_exist(image_id)
            if not image_id_exist and wait:
                self.task_manager.tasks[task_id].image_id_queue.put(image_id)
            # 设置超时时间为 1 秒
            timeout = 1
            start_time = time.time()

            # 等待检测完成
            while detector.get_statue(image_id) != 0:
                # 检查是否超过了超时时间
                if time.time() - start_time > timeout:
                    raise TimeoutError("等待超时")
                
                time.sleep(0.01)
            
            results = detector.get_result_by_uid(image_id)
        except Exception as e:
            response_code = 400
            response_message += str(e)
            print(e)

        response = target_detection_pb2.GetResultIndexByImageIdResponse()
        response.response.code = response_code
        response.response.message = response_message
        for result in results:
            result_response = target_detection_pb2.Result()
            x1, y1, x2, y2, c, conf = result
            result_response.labelId = c
            result_response.confidence = conf
            result_response.x1 = x1
            result_response.y1 = y1
            result_response.x2 = x2
            result_response.y2 = y2
            response.results.append(result_response)
        return response

    def getLatestResultIndex(self, request, context):
        response_code = 200
        response_message = ''
        results = []
        image_id = 0
        try:
            task_id = request.taskId
            assert task_id in self.task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            weight = self.task_manager.tasks[task_id].weight
            assert weight in self.detector_manager.detector_info_map, f'ERROR: can not find {weight} in detector_info_map.\n'
            detector = self.detector_manager.detector_info_map[weight].detector
            image_id = detector.latest_detection_completed_uid
            results = detector.get_result_by_uid(image_id)
            # TODO results为None时会出bug
        except Exception as e:
            response_code = 400
            response_message += str(e)

        response = target_detection_pb2.GetLatestResultIndexResponse()
        response.response.code = response_code
        response.response.message = response_message
        response.imageId = image_id
        for result in results:
            result_response = target_detection_pb2.Result()
            x1, y1, x2, y2, c, conf = result
            result_response.labelId = c
            result_response.confidence = conf
            result_response.x1 = x1
            result_response.y1 = y1
            result_response.x2 = x2
            result_response.y2 = y2
            response.results.append(result_response)
        return response
    
    def join_in_server(self, server):
        target_detection_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
