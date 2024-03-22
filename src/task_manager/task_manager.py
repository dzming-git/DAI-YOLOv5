from src.utils import singleton
from typing import Dict, List
import queue
from src.grpc.clients.image_harmony.image_harmony_client import ImageHarmonyClient
from src.wrapper.yolov5_detector import YOLOv5Detector
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TaskInfo:
    def __init__(self, task_id: int):
        self.id: int = task_id
        
        self.__image_harmony_address: List[str, str] = []
        self.__image_harmony_client: ImageHarmonyClient = None
        
        self.__loader_args_hash: int = 0  # image harmony中加载器的hash值
        self.weight: str = ''
        self.__device_str: str = ''
        self.image_id_queue: queue.Queue[int] = queue.Queue()
        self.detector: YOLOv5Detector = None
        self.__stop_event = threading.Event()
        self.__track_thread = None  # 用于跟踪线程的引用
    
    def set_pre_service(self, pre_service_name: str, pre_service_ip: str, pre_service_port: str, args: Dict[str, str] = {}):
        if 'image harmony gRPC' == pre_service_name:
            self.__image_harmony_address = [pre_service_ip, pre_service_port]
            self.__image_harmony_client = ImageHarmonyClient(pre_service_ip, pre_service_port)
            if 'LoaderArgsHash' not in args:
                raise ValueError('Argument "LoaderArgsHash" is required but not set.')
            self.__loader_args_hash = int(args['LoaderArgsHash'])
    
    def set_cur_service(self, args: Dict[str, str]):
        # TODO 未来添加dnn half等
        if 'Device' in args:
            self.__device_str = args['Device']
        if 'Weight' in args:
            self.weight = args['Weight']
    
    def check(self) -> None:
        if not self.__image_harmony_address:
            raise ValueError('Error: image_harmony_address not set.')
        if not self.__image_harmony_client:
            raise ValueError('Error: image_harmony_client not set.')
        if not self.__loader_args_hash:
            raise ValueError('Error: loader_args_hash not set.')
        if not self.weight:
            raise ValueError('Error: weight not set.')
        if not self.__device_str:
            raise ValueError('Error: device not set.')

    def start(self):
        self.check()
        self.__image_harmony_client.connect_image_loader(self.__loader_args_hash)
        yolov5_builder = YOLOv5Detector.YOLOv5Builder()
        yolov5_builder.weight = self.weight
        yolov5_builder.device_str = self.__device_str
        self.detector = yolov5_builder.build()
        self.detector.load_model()
        self.__stop_event.clear()  # 确保开始时事件是清除状态
        self.__track_thread = threading.Thread(target=self.detect_by_image_id)
        self.__track_thread.start()
    
    def detect_by_image_id(self):
        while not self.__stop_event.is_set():  # 使用事件来检查停止条件
            # image_id_in_queue = self.image_id_queue.get()
            try:
            # 尝试从队列中获取image_id，设置超时时间为1秒
                image_id_in_queue = self.image_id_queue.get(timeout=1)
            except queue.Empty:
                # 如果在超时时间内没有获取到新的image_id，则继续循环，此时可以检查停止事件
                continue
            try:
                width, height = self.__image_harmony_client.get_image_size_by_image_id(image_id_in_queue)
                new_unpad_width, new_unpad_height, top, bottom, left, right = self.detector.get_letterbox_size(width, height)
                image_id, image = self.__image_harmony_client.get_image_by_image_id(image_id_in_queue, new_unpad_width, new_unpad_height)
                if 0 == image_id:
                    continue
                self.detector.add_image(image_id, image)
                self.detector.detect_by_image_id(image_id)
            except Exception as e:
                logging.error(e)
                
    
    def stop(self):
        self.__stop_event.set()  # 设置事件，通知线程停止
        if self.__track_thread:
            self.__track_thread.join()  # 等待线程结束
        self.__image_harmony_client.disconnect_image_loader()
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