from src.utils import singleton
from typing import Dict
from queue import Queue
from src.grpc.img_trans.img_trans_client import ImgTransClient
import _thread
from yolov5_src import YOLOV5Impl

class TaskInfo:
    def __init__(self):
        self.id: int
        self.stop: bool
        self.pre_service_name: str
        self.pre_service_ip: str
        self.pre_service_port: str
        self.img_type: str
        self.img_args: Dict[str, str] = {}
        self.img_trans_client: ImgTransClient = None


@singleton
class TaskCtrl:
    def __init__(self):
        self.tasks_queue: Queue[TaskInfo] = Queue(maxsize=20)
        self.tasks: Dict[int, TaskInfo] = {}
        self.yolov5_impl: YOLOV5Impl = None
    
    def set_yolov5_impl(self, impl: YOLOV5Impl):
        self.yolov5_impl = impl

    def listening(self):
        def wait_for_task():
            while True:
                task = self.tasks_queue.get()
                task_id = task.id
                if task_id not in self.tasks:
                    task.img_trans_client = ImgTransClient(task.pre_service_ip, task.pre_service_port)
                    task.img_trans_client.set_args(task.img_type, task.img_args)
                    task.stop = False
                    self.tasks[task_id] = task

                    _thread.start_new_thread(self.progress, (task_id,))

        _thread.start_new_thread(wait_for_task, ())

    def progress(self, task_id: int):
        assert self.yolov5_impl is not None, 'yolov5 impl is not set\n'
        while not self.tasks[task_id].stop:
            img_id, img = self.tasks[task_id].img_trans_client.get_img()
            self.yolov5_impl.add_img(img_id, img)
            self.yolov5_impl.detect_by_uid(img_id)
            result = self.yolov5_impl.get_result_by_uid(img_id)
            if (result):
                print(result)
