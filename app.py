from yolov5_src import YOLOV5Impl
from grpcs.task_coordinate.task_coordinate_server import task_coordinate_serve
from utils.task_ctrl import TaskCtrl

if __name__ == '__main__':
    yolov5_builder = YOLOV5Impl.YOLOV5Builder()
    yolov5_builder._device = 'cuda:0'
    yolov5_impl = yolov5_builder.build()
    yolov5_impl.load_model()

    task_ctrl = TaskCtrl()
    task_ctrl.set_yolov5_impl(yolov5_impl)
    task_ctrl.listening()
    task_coordinate_serve('0.0.0.0', '5000', 10)
