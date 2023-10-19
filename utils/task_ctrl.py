from common.utils import singleton
from typing import Dict
from queue import Queue
from grpcs.img_trans.img_trans_client import ImgTransClient
import _thread

class TaskInfo:
    def __init__(self):
        self.id: int
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
    
    def listening(self):
        def wait_for_task():
            while True:
                task = self.tasks_queue.get()
                task_id = task.id
                if task_id not in self.tasks:
                    task.img_trans_client = ImgTransClient(task.pre_service_ip, task.pre_service_port)
                    task.img_trans_client.set_args(task.img_type, task.img_args)
                    self.tasks[task_id] = task

        _thread.start_new_thread(wait_for_task, ())
