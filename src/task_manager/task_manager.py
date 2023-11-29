from src.utils import singleton
from typing import Dict
from queue import Queue
from src.grpc.clients.image_harmony.image_harmony_client import ImageHarmonyClient
import _thread
from src.wrapper.yolov5_detector import YOLOv5Detector
from src.detector_manager.detector_manager import DetectorManager, DetectorInfo

class TaskInfo:
    def __init__(self):
        self.id: int
        self.stop: bool
        self.pre_service_name: str
        self.pre_service_ip: str
        self.pre_service_port: str
        self.image_harmony_client: ImageHarmonyClient = None
        self.connect_id: int  # 与image harmony连接的id，根据该id获取图像
        self.weight: str
        self.is_pre_service_set = False
        self.is_detector_set = False
        self.step = 2
        self.image_id_queue: Queue[int] = Queue()
    
    def set_pre_service(self, pre_service_name: str, pre_service_ip: str, pre_service_port: str):
        self.pre_service_name = pre_service_name
        self.pre_service_ip = pre_service_ip
        self.pre_service_port = pre_service_port
        if not self.is_pre_service_set:
            self.is_pre_service_set = True
            self.step -= 1
    
    def set_cur_service(self, weight: str, connect_id: int):
        self.weight = weight
        self.connect_id = connect_id
        if not self.is_detector_set:
            self.is_detector_set = True
            self.step -= 1
    
    def start(self):
        # 设置图像源
        self.image_harmony_client = ImageHarmonyClient(self.pre_service_ip, self.pre_service_port)
        self.image_harmony_client.set_connect_id(self.connect_id)
        # 加载权重
        detector_manager = DetectorManager()
        # TODO 暂时使用weight作为哈希值，未来将检测器的其他属性也考虑进去
        if self.weight not in detector_manager.detector_info_map:
            yolov5_builder = YOLOv5Detector.YOLOv5Builder()
            yolov5_builder.weight_path = f'./weights/{self.weight}'
            detector = yolov5_builder.build()
            detector.load_model()
            if self.weight not in detector_manager.detector_info_map:
                detector_manager.detector_info_map[self.weight] = DetectorInfo()
            # TODO 需要添加一个权重加载成功的判断
            detector_manager.detector_info_map[self.weight].weight_path = yolov5_builder.weight_path
            detector_manager.detector_info_map[self.weight].detector = detector
            detector_manager.detector_info_map[self.weight].cnt += 1
        
        self.stop = False
        # _thread.start_new_thread(self.progress, ())
        # TODO 临时版本
        _thread.start_new_thread(self.detect_by_image_id, ())

    def progress(self):
        # TODO 两种方案，延迟？性能？目前采用的是延迟最低的方案
        # 延迟最低的方案，申请最新的图片，持续检测，根据图片ID查询结果
        # 另一种方案不浪费性能，接收图片ID，根据图片ID去申请图片检测
        detector_manager = DetectorManager()
        assert self.weight in detector_manager.detector_info_map and detector_manager.detector_info_map[self.weight] is not None, 'yolov5 detector is not set\n'
        assert self.image_harmony_client is not None, 'image harmony client is not set\n'
        detector_info = detector_manager.detector_info_map[self.weight]
        while not self.stop:
            # TODO 优化：先获取图像id，判断是否重复，不重复再用id获取图像
            image_id, img = self.image_harmony_client.get_latest_image()
            if 0 == image_id:
                continue
            if not detector_info.detector.add_img(image_id, img):
                continue
            detector_info.detector.detect_by_uid(image_id)
            result = detector_info.detector.get_result_by_uid(image_id)
            if (result):
                print(result)
    
    def detect_by_image_id(self):
        detector_manager = DetectorManager()
        assert self.weight in detector_manager.detector_info_map and detector_manager.detector_info_map[self.weight] is not None, 'yolov5 detector is not set\n'
        assert self.image_harmony_client is not None, 'image harmony client is not set\n'
        detector_info = detector_manager.detector_info_map[self.weight]
        while not self.stop:
            image_id_in_queue = self.image_id_queue.get()
            image_id, image = self.image_harmony_client.get_image_by_image_id(image_id_in_queue)
            if 0 == image_id:
                continue
            if not detector_info.detector.add_img(image_id, image):
                continue
            detector_info.detector.detect_by_uid(image_id)
            result = detector_info.detector.get_result_by_uid(image_id)
            if (result):
                print(result)
                                                                                                                                           
@singleton
class TaskManager:
    def __init__(self):
        self.tasks_queue: Queue[TaskInfo] = Queue(maxsize=20)
        self.incomplete_tasks: Dict[int, TaskInfo] = {}
        self.tasks: Dict[int, TaskInfo] = {}
    
    def set_detector(self, task_id: int, detector: YOLOv5Detector):
        self.tasks[task_id].detector = detector

    def listening(self):
        def wait_for_task():
            while True:
                task = self.tasks_queue.get()
                task.start()

        _thread.start_new_thread(wait_for_task, ())
