from generated.protos.target_detection import target_detection_pb2, target_detection_pb2_grpc
import time
import traceback
from src.task_manager.task_manager import TaskManager
from src.model_manager.model_manager import ModelManager
from src.model_manager import model_manager as mm
from src.config.config import Config

task_manager = TaskManager()
model_manager = ModelManager()
config = Config()

class TargetDetectionServer(target_detection_pb2_grpc.CommunicateServicer):
    def getResultMappingTable(self, request, context):
        response_code = 200
        response_message = ''
        response = target_detection_pb2.GetResultMappingTableResponse()
        labels = []
        try:
            task_id = request.taskId
            if task_id not in task_manager.tasks:
                raise ValueError('ERROR: The task ID does not exist.')
            weight = task_manager.tasks[task_id].weight
            if weight not in config.weights_map:
                raise ValueError(f'ERROR: The configuration file does not contain the weight: {weight}.')
            weight_info = config.weights_map[weight]
            labels = weight_info.labels
        except Exception as e:
            response.response.code = 400
            response.response.message = str(e)
            return response

        response.response.code = response_code
        response.response.message = response_message
        for label in labels:
            response.labels.append(label)
        return response
    
    def loadModel(self, request, context):
        response_code = 200
        response_message = ''
        response = target_detection_pb2.LoadModelResponse()
        try:
            task_id = request.taskId
            if task_id not in task_manager.tasks:
                raise ValueError('ERROR: The task ID does not exist.')
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
        except Exception as e:
            response.response.code = 400
            response.response.message = str(e)
            return response
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def checkModelState(self, request, context):
        response_code = 200
        response_message = ''
        response = target_detection_pb2.CheckModelStateResponse()
        modelState = target_detection_pb2.ModelState.NotSet
        try:
            task_id = request.taskId
            if task_id not in task_manager.tasks:
                raise ValueError('ERROR: The task ID does not exist.')
            detector = task_manager.tasks[task_id].detector
            weight = detector._weight
            device = detector._device
            dnn = detector._dnn
            half = detector._half
            model_info = model_manager.get_model_info(weight, device, dnn, half)
            model_state = model_info.get_state()
            if model_state == mm.NOT_LOADED:
                modelState = target_detection_pb2.ModelState.NotLoaded
            elif model_state == mm.LOADING:
                modelState = target_detection_pb2.ModelState.Loading
            elif model_state == mm.LOADING_COMPLETED:
                modelState = target_detection_pb2.ModelState.LoadingCompleted
        except Exception as e:
            response.response.code = 400
            response.response.message = str(e)
            return response

        response.response.code = response_code
        response.response.message = response_message
        response.modelState = modelState
        return response

    def getResultIndexByImageId(self, request, context):
        response_code = 200
        response_message = ''
        response = target_detection_pb2.GetResultIndexByImageIdResponse()
        results = []
        try:
            task_id = request.taskId
            image_id = request.imageId
            wait = request.wait
            if task_id not in task_manager.tasks:
                raise ValueError('ERROR: The task ID does not exist.')
            detector = task_manager.tasks[task_id].detector
            image_id_exist = detector.check_image_id_exist(image_id)
            if not image_id_exist and wait:
                task_manager.tasks[task_id].image_id_queue.put(image_id)
            # 设置超时时间为 1 秒
            timeout = 1
            start_time = time.time()
            
            # 等待添加完成
            while not detector.check_image_id_exist(image_id):
                # 检查是否超过了超时时间
                if time.time() - start_time > timeout:
                    raise TimeoutError("添加图片超时")
                time.sleep(0.01)
            
            # 等待检测完成
            while detector.get_status(image_id) != 0:
                # 检查是否超过了超时时间
                if time.time() - start_time > timeout:
                    raise TimeoutError("检测图片超时")
                time.sleep(0.01)
            
            results = detector.get_result_by_image_id(image_id)
        except Exception as e:
            response.response.code = 400
            response.response.message = str(e)
            return response

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
        response = target_detection_pb2.GetLatestResultIndexResponse()
        results = []
        image_id = 0
        try:
            task_id = request.taskId
            if task_id not in task_manager.tasks:
                raise ValueError('ERROR: The task ID does not exist.')
            detector = task_manager.tasks[task_id].detector
            image_id = detector.latest_detection_completed_image_id
            results = detector.get_result_by_image_id(image_id)
            # TODO results为None时会出bug
        except Exception as e:
            response.response.code = 400
            response.response.message = str(e)
            return response

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
