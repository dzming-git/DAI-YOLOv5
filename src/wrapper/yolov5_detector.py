import numpy as np
import torch
import cv2
from yolov5_src.utils.general import non_max_suppression, scale_boxes, check_img_size
from yolov5_src.utils.plots import Annotator, colors
from yolov5_src.utils.augmentations import letterbox
from typing import Dict
import queue
import copy
import threading
import traceback
from src.model_manager.model_manager import ModelManager
import warnings
warnings.filterwarnings('always')

# step说明
# 0 完成检测图片
# 1 开始检测图片
# 2 完成添加图片
# 3 开始添加图片
# 4 完成添加uid
DETECT_IMAGE_COMPLETE = 0
DETECT_IMAGE_START = 1
ADD_IMAGE_COMPLETE = 2
ADD_IMAGE_START = 3
ADD_UID_COMPLETE = 4

model_manager = ModelManager()

class YOLOv5Detector:
    class YOLOv5Builder:
        def __init__(self):
            self.weight = 'yolov5s.pt'
            self.device = 'cpu'
            self.imgsz = [640] * 2
            self.conf_thres = 0.5
            self.iou_thres = 0.45
            self.max_det = 1000
            self.agnostic_nms = False
            self.augment = False
            self.half = False
            self.dnn = False

            # 是否将缓存图像，默认不缓存
            self.save_img_to_cache = False

            # 最大缓存100个图像结果
            self.max_cache = 100

            # get_result_by_uid 是否打印结果
            self.print_result = False
        
        def build(self):
            if not torch.cuda.is_available():
                if self.device != 'cpu':
                    warnings.warn("cuda is not available", UserWarning)
                self.device = 'cpu'
            # self.device 字符串转torch.device
            try:
                self.device = torch.device(self.device)
            except Exception as e:
                traceback.print_exc()
                self.device = torch.device('cpu')
            return YOLOv5Detector(self)

    class ImgInfo:
        uid = 0
        img = None
        img_shape = np.zeros((0, 0, 0))
        img_processed = None
        pred = np.zeros((0, 0, 0))
        is_used = False
        step = ADD_UID_COMPLETE
        lock = threading.Lock()

    def __init__(self, builder:YOLOv5Builder):
        self._weight = builder.weight
        self._device = builder.device
        self._imgsz = builder.imgsz
        self._conf_thres = builder.conf_thres
        self._iou_thres = builder.iou_thres
        self._max_det = builder.max_det
        self._agnostic_nms = builder.agnostic_nms
        self._augment = builder.augment
        self._half = builder.half & (self._device.type != 'cpu')
        self._dnn = builder.dnn
        self._max_cache = builder.max_cache

        # 是否打印
        self._print_result = builder.print_result

        # 延迟加载
        self._model_info = model_manager.get_model_info(self._weight, self._device, self._dnn, self._half)

        # 图像、结果缓存
        self._save_img_to_cache = builder.save_img_to_cache
        self._img_infos:Dict[int, YOLOv5Detector.ImgInfo] = dict()
        self._img_uid_fifo:queue.Queue[int] = queue.Queue(maxsize=self._max_cache)
        
        # 最新检测完成的uid
        self.latest_detection_completed_uid = 0
        # 最新添加图片的uid
        self.latest_add_uid = 0

    def load_model(self) -> bool:
        is_load_success = self._model_info.start_using()
        if is_load_success:
            self._imgsz = check_img_size(self._imgsz, s=self._model_info._model.stride)  # check image size
        return is_load_success

    def check_uid_exist(self, uid) -> bool:
        return uid in self._img_infos
    
    def get_statue(self, uid) -> int:
        if not self.check_uid_exist(uid):
            # warnings.warn("该uid不存在", UserWarning)
            return -1
        return self._img_infos[uid].step

    def add_uid(self, uid) -> bool:
        if self.check_uid_exist(uid):
            warnings.warn("该uid已存在", UserWarning)
            return False
        # 清理溢出
        if (len(self._img_infos) >= self._max_cache):
            uid_rm = self._img_uid_fifo.get()
            if not self._img_infos[uid_rm].is_used:
                warnings.warn(f'弹出uid={uid_rm}-未被使用', UserWarning)
            with self._img_infos[uid_rm].lock:
                self._img_infos.pop(uid_rm)
        self._img_uid_fifo.put(uid)
        self._img_infos[uid] = YOLOv5Detector.ImgInfo()
        self._img_infos[uid].step = ADD_UID_COMPLETE
        return True

    @torch.no_grad()
    def detect_by_uid(self, uid) -> bool:
        if not self.check_uid_exist(uid):
            warnings.warn("该uid不存在", UserWarning)
            return False
        if self._img_infos[uid].step != ADD_IMAGE_COMPLETE:
            warnings.warn("图片未添加完成", UserWarning)
            return False
        self._img_infos[uid].step = DETECT_IMAGE_START
        with self._img_infos[uid].lock:
            pred = self._model_info._model(self._img_infos[uid].img_processed)[0]
        pred = non_max_suppression(prediction=pred,
                                          conf_thres=self._conf_thres,
                                          iou_thres=self._iou_thres,
                                          max_det=self._max_det)[0]
        with self._img_infos[uid].lock:
            if len(pred):
                pred[:, :4] = scale_boxes(self._img_infos[uid].img_processed.shape[2:], pred[:, :4], self._img_infos[uid].img_shape).round()
            self._img_infos[uid].pred = pred
        
        del self._img_infos[uid].img_processed
        self._img_infos[uid].img_processed = None
        self.latest_detection_completed_uid = uid
        self._img_infos[uid].step = DETECT_IMAGE_COMPLETE

    def get_result_by_uid(self, uid):
        result = []
        if not self.check_uid_exist(uid):
            warnings.warn("该uid不存在", UserWarning)
            return None
        if self._img_infos[uid].step != DETECT_IMAGE_COMPLETE:
            warnings.warn("图片未添加完成", UserWarning)
            return False
        with self._img_infos[uid].lock:
            self._img_infos[uid].is_used = True
            if len(self._img_infos[uid].pred):
                for *xyxy, conf, cls in reversed(self._img_infos[uid].pred):
                    c = int(cls)  # integer class
                    label = f'{self._model_info._model.names[c]}'
                    imgRows, imgCols, _ = self._img_infos[uid].img_shape
                    result.append([
                        xyxy[0].item() / imgCols,
                        xyxy[1].item() / imgRows,
                        xyxy[2].item() / imgCols,
                        xyxy[3].item() / imgRows,
                        c,
                        float(conf.cpu().numpy())
                    ])
                    if self._print_result is True:
                        print('YoloImgInterface', result[-1])
        return result

    def get_imglabeled_by_uid(self, uid):
        if not self._save_img_to_cache:
            warnings.warn("save_img_to_cache is False", UserWarning)
            return cv2.Mat()

        with self._img_infos[uid].lock:
            annotator = Annotator(self._img_infos[uid].img, line_width=3, example=str(self._model_info._model.names))
        self._img_infos[uid].is_used = True
        if len(self._img_infos[uid].pred):
            for *xyxy, conf, cls in reversed(self._img_infos[uid].pred):
                c = int(cls)  # integer class
                annotator.box_label(xyxy, f'{self._model_info._model.names[c]} {conf:.2f}', color=colors(c, True))
        img = annotator.result()
        return img

    def add_img(self, uid, img) -> bool:
        if not self.check_uid_exist(uid):
            self.add_uid(uid)
        elif self._img_infos[uid].step != ADD_UID_COMPLETE:
            print(self._img_infos[uid].step)
            warnings.warn("重复添加", UserWarning)
            return False
        self._img_infos[uid].step = ADD_IMAGE_START
        self._img_infos[uid].img_shape = img.shape
        if self._save_img_to_cache:
            self._img_infos[uid].img = copy.deepcopy(img)
        # 处理图片
        img = letterbox(img, self._imgsz, stride=self._model_info._model.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        self._img_infos[uid].img_processed = copy.deepcopy(img)
        self.latest_add_uid = uid
        self._img_infos[uid].step = ADD_IMAGE_COMPLETE
        return True
