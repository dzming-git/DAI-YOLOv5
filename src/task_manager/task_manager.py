from src.utils import singleton
from typing import Dict, Tuple, List
import queue
from src.grpc.clients.image_harmony.image_harmony_client import ImageHarmonyClient
from src.wrapper.yolov5_detector import YOLOv5Detector
import traceback
import threading

class TaskInfo:
    def __init__(self, task_id: int):
        self.id: int = task_id
        
        self.image_harmony_address: List[str, str] = []
        self.image_harmony_client: ImageHarmonyClient = None
        
        self.loader_args_hash: int = 0  # image harmony中加载器的hash值
        self.weight: str = ''
        self.device: str = ''
        self.image_id_queue: queue.Queue[int] = queue.Queue()
        self.detector: YOLOv5Detector = None
        self.stop_event = threading.Event()
        self.track_thread = None  # 用于跟踪线程的引用
    
    def set_pre_service(self, pre_service_name: str, pre_service_ip: str, pre_service_port: str, args: Dict[str, str] = {}):
        if 'image harmony gRPC' == pre_service_name:
            self.image_harmony_address = [pre_service_ip, pre_service_port]
            self.image_harmony_client = ImageHarmonyClient(pre_service_ip, pre_service_port)
            assert 'LoaderArgsHash' in args, 'arg: [LoaderArgsHash] not set'
            self.loader_args_hash = int(args['LoaderArgsHash'])
    
    def set_cur_service(self, args: Dict[str, str]):
        # TODO 未来添加dnn half等
        if 'Device' in args:
            self.device = args['Device']
        if 'Weight' in args:
            self.weight = args['Weight']
    
    def check(self) -> Tuple[bool, str]:
        try:
            assert self.image_harmony_address, 'Error: image_harmony_address not set.'
            assert self.image_harmony_client,  'Error: image_harmony_client not set.'
            assert self.loader_args_hash,      'Error: loader_args_hash not set.'
            assert self.weight,                'Error: weight not set.'
            assert self.device,                'Error: device not set.'
        except Exception as e:
            error_info = traceback.format_exc()
            return False, error_info
        return True, 'OK'
    
    def start(self):
        try:
            self.image_harmony_client.connect_image_loader(self.loader_args_hash)
            yolov5_builder = YOLOv5Detector.YOLOv5Builder()
            yolov5_builder.weight = self.weight
            yolov5_builder.device_str = self.device
            self.detector = yolov5_builder.build()
            assert self.detector.load_model(), 'load model failed'
            self.stop_event.clear()  # 确保开始时事件是清除状态
            self.track_thread = threading.Thread(target=self.detect_by_image_id)
            self.track_thread.start()
        except Exception as e:
            error_info = traceback.format_exc()
            print(error_info)
            return False, error_info
        return True, 'OK'
    # def progress(self):
    #     # TODO 两种方案，延迟？性能？目前采用的是延迟最低的方案
    #     # 延迟最低的方案，申请最新的图片，持续检测，根据图片ID查询结果
    #     # 另一种方案不浪费性能，接收图片ID，根据图片ID去申请图片检测
    #     detector_manager = DetectorManager()
    #     assert self.weight in detector_manager.detector_info_map and detector_manager.detector_info_map[self.weight] is not None, 'yolov5 detector is not set\n'
    #     assert self.image_harmony_client is not None, 'image harmony client is not set\n'
    #     detector_info = detector_manager.detector_info_map[self.weight]
    #     while not self.stop:
    #         # TODO 优化：先获取图像id，判断是否重复，不重复再用id获取图像
    #         image_id, img = self.image_harmony_client.get_latest_image()
    #         if 0 == image_id:
    #             continue
    #         if not detector_info.detector.add_img(image_id, img):
    #             continue
    #         detector_info.detector.detect_by_image_id(image_id)
    #         result = detector_info.detector.get_result_by_image_id(image_id)
    #         if (result):
    #             print(result)
    
    def detect_by_image_id(self):
        assert self.detector, 'yolov5 detector is not set\n'
        assert self.image_harmony_client, 'image harmony client is not set\n'
        while not self.stop_event.is_set():  # 使用事件来检查停止条件
            # image_id_in_queue = self.image_id_queue.get()
            try:
            # 尝试从队列中获取image_id，设置超时时间为1秒
                image_id_in_queue = self.image_id_queue.get(timeout=1)
            except queue.Empty:
                # 如果在超时时间内没有获取到新的image_id，则继续循环，此时可以检查停止事件
                continue
            width, height = self.image_harmony_client.get_image_size_by_image_id(image_id_in_queue)
            if 0 == width or 0 == height:
                continue
            new_unpad_width, new_unpad_height, top, bottom, left, right = self.detector.get_letterbox_size(width, height)
            image_id, image = self.image_harmony_client.get_image_by_image_id(image_id_in_queue, new_unpad_width, new_unpad_height)
            if 0 == image_id:
                continue
            if not self.detector.add_image(image_id, image):
                continue
            self.detector.detect_by_image_id(image_id)
    
    def stop(self):
        self.stop_event.set()  # 设置事件，通知线程停止
        if self.track_thread:
            self.track_thread.join()  # 等待线程结束
        self.image_harmony_client.disconnect_image_loader()
        if self.detector:
            del self.detector  # 释放资源
        self.detector = None                                                                                                                                    

@singleton
class TaskManager:
    def __init__(self):
        self.tasks: Dict[int, TaskInfo] = {}
        self.__lock = threading.Lock()
    
    def stop_task(self, task_id: int):
        with self.__lock:
            if task_id in self.tasks:
                self.tasks[task_id].stop()
                del self.tasks[task_id]