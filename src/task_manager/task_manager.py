from src.utils import singleton
from typing import Dict
from queue import Queue
from src.grpc.clients.image_harmony.image_harmony_client import ImageHarmonyClient
import _thread
from src.wrapper.yolov5_detector import YOLOv5Detector
from src.detector_manager.detector_manager import DetectorManager

class TaskInfo:
    def __init__(self):
        self.id: int
        self.stop: bool
        self.pre_service_name: str
        self.pre_service_ip: str
        self.pre_service_port: str
        self.image_harmony_client: ImageHarmonyClient = None
        self.connect_id: int  # 与image harmony连接的id，根据该id获取图像
        self.weight_path: str
        self.is_pre_service_set = False
        self.is_detector_set = False
        self.step = 2
    
    def set_pre_service(self, pre_service_name: str, pre_service_ip: str, pre_service_port: str):
        self.pre_service_name = pre_service_name
        self.pre_service_ip = pre_service_ip
        self.pre_service_port = pre_service_port
        if not self.is_pre_service_set:
            self.is_pre_service_set = True
            self.step -= 1
    
    def set_cur_service(self, weight_path: str, connect_id: int):
        self.weight_path = weight_path
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
        # TODO 暂时使用weight_path作为哈希值，未来将检测器的其他属性也考虑进去
        if self.weight_path not in detector_manager.detector_map:
            yolov5_builder = YOLOv5Detector.YOLOv5Builder()
            yolov5_builder.weights = self.weight_path
            detector = yolov5_builder.build()
            detector.load_model()
            detector_manager.detector_map[self.weight_path] = detector
        
        self.stop = False
        _thread.start_new_thread(self.progress, ())

    def progress(self):
        detector_manager = DetectorManager()
        assert self.weight_path in detector_manager.detector_map and detector_manager.detector_map[self.weight_path] is not None, 'yolov5 detector is not set\n'
        assert self.image_harmony_client is not None, 'image harmony client is not set\n'
        detector = detector_manager.detector_map[self.weight_path]
        while not self.stop:
            img_id, img = self.image_harmony_client.get_img()
            if 0 == img_id:
                continue
            detector.add_img(img_id, img)
            detector.detect_by_uid(img_id)
            result = detector.get_result_by_uid(img_id)
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
