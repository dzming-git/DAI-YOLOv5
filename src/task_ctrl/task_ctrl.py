from src.utils import singleton
from typing import Dict
from queue import Queue
from src.grpc.clients.image_harmony.image_harmony_client import ImageHarmonyClient
import _thread
from src.wrapper.yolov5_detector import YOLOv5Detector

class TaskInfo:
    def __init__(self):
        self.id: int
        self.stop: bool
        self.pre_service_name: str
        self.pre_service_ip: str
        self.pre_service_port: str
        self.img_type: str
        self.img_args: Dict[str, str] = {}
        self.image_harmony_client: ImageHarmonyClient = None


@singleton
class TaskCtrl:
    def __init__(self):
        self.tasks_queue: Queue[TaskInfo] = Queue(maxsize=20)
        self.tasks: Dict[int, TaskInfo] = {}
        self.detector: YOLOv5Detector = None
    
    def set_detector(self, detector: YOLOv5Detector):
        self.detector = detector

    def listening(self):
        def wait_for_task():
            while True:
                task = self.tasks_queue.get()
                task_id = task.id
                if task_id not in self.tasks:
                    task.image_harmony_client = ImageHarmonyClient(task.pre_service_ip, task.pre_service_port)
                    task.image_harmony_client.set_args(task.img_type, task.img_args)
                    task.stop = False
                    self.tasks[task_id] = task

                    _thread.start_new_thread(self.progress, (task_id,))

        _thread.start_new_thread(wait_for_task, ())

    def progress(self, task_id: int):
        assert self.detector is not None, 'yolov5 detector is not set\n'
        while not self.tasks[task_id].stop:
            img_id, img = self.tasks[task_id].image_harmony_client.get_img()
            self.detector.add_img(img_id, img)
            self.detector.detect_by_uid(img_id)
            result = self.detector.get_result_by_uid(img_id)
            if (result):
                print(result)
