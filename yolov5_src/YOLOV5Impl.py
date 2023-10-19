import numpy as np
import torch
from yolov5_src.utils.general import non_max_suppression, scale_boxes, check_img_size
from yolov5_src.utils.plots import Annotator, colors
from yolov5_src.utils.augmentations import letterbox
from yolov5_src.models.common import DetectMultiBackend
from yolov5_src.utils.torch_utils import select_device
from typing import Dict
import queue
import copy
import threading
import warnings
warnings.filterwarnings('always')


class YOLOV5Impl:
    class YOLOV5Builder:
        def __init__(self):
            self._weights = 'weights/yolov5s.pt'
            self._device = 'cpu'
            self._imgsz = [640] * 2
            self._conf_thres = 0.5
            self._iou_thres = 0.45
            self._max_det = 1000
            self._agnostic_nms = False
            self._augment = False
            self._half = False
            self._dnn = False
            self._data = 'yolov5_src/data/coco128.yaml'

            # 最大缓存10个图像结果
            self._max_cache = 10

            # get_result_by_uid 是否打印结果
            self._print_result = False
        
        def build(self):
            if not torch.cuda.is_available():
                if self._device != 'cpu':
                    warnings.warn("cuda is not available", UserWarning)
                self._device = 'cpu'
            return YOLOV5Impl(self)

    class ImgInfo:
        uid = -1
        img = None
        imgProcessed = None
        pred = np.zeros((0, 0, 0))
        is_used = False
        lock = threading.Lock()

    def __init__(self, builder:YOLOV5Builder):
        self._weights = builder._weights
        self._device = select_device(builder._device)
        self._imgsz = builder._imgsz
        self._conf_thres = builder._conf_thres
        self._iou_thres = builder._iou_thres
        self._max_det = builder._max_det
        self._agnostic_nms = builder._agnostic_nms
        self._augment = builder._augment
        self._half = builder._half & (self._device.type != 'cpu')
        self._dnn = builder._dnn
        self._data = builder._data
        self._max_cache = builder._max_cache

        # 是否打印
        self._print_result = builder._print_result

        # 延迟加载
        self._model = None
        self._model_load_lock = threading.Lock()

        # 图像、结果缓存
        self._img_infos:Dict[int, YOLOV5Impl.ImgInfo] = dict()
        self._img_uid_fifo:queue.Queue[int] = queue.Queue(maxsize=self._max_cache)


    def load_model(self) -> bool:
        is_load_success = False
        if self._model is None:
            with self._model_load_lock:
                if self._model is None:
                    try:
                        self._model = DetectMultiBackend(self._weights, device=self._device, dnn=self._dnn, data=self._data, fp16=self._half)
                        self._imgsz = check_img_size(self._imgsz, s=self._model.stride)  # check image size
                        is_load_success = True
                    except Exception as e:
                        warnings.warn(e, UserWarning)
        return is_load_success

    @torch.no_grad()
    def detect_by_uid(self, uid) -> bool:
        if uid not in self._img_infos:
            warnings.warn("该uid不存在", UserWarning)
            return False
        with self._img_infos[uid].lock:
            pred = self._model(self._img_infos[uid].imgProcessed)[0]
        pred = non_max_suppression(prediction=pred,
                                          conf_thres=self._conf_thres,
                                          iou_thres=self._iou_thres,
                                          max_det=self._max_det)[0]
        with self._img_infos[uid].lock:
            if len(pred):
                pred[:, :4] = scale_boxes(self._img_infos[uid].imgProcessed.shape[2:], pred[:, :4], self._img_infos[uid].img.shape).round()
            self._img_infos[uid].pred = pred

    def get_result_by_uid(self, uid):
        result = []
        with self._img_infos[uid].lock:
            self._img_infos[uid].is_used = True
            if len(self._img_infos[uid].pred):
                for *xyxy, conf, cls in reversed(self._img_infos[uid].pred):
                    c = int(cls)  # integer class
                    label = f'{self._model.names[c]}'
                    imgRows, imgCols, _ = self._img_infos[uid].img.shape
                    result.append([
                        xyxy[0].item() / imgCols,
                        xyxy[1].item() / imgRows,
                        xyxy[2].item() / imgCols,
                        xyxy[3].item() / imgRows,
                        [[label, float(conf.cpu().numpy())]]
                    ])
                    if self._print_result is True:
                        print('YoloImgInterface', result[-1])
        return result

    def get_imglabeled_by_uid(self, uid):
        with self._img_infos[uid].lock:
            annotator = Annotator(self._img_infos[uid].img, line_width=3, example=str(self._model.names))
        self._img_infos[uid].is_used = True
        if len(self._img_infos[uid].pred):
            for *xyxy, conf, cls in reversed(self._img_infos[uid].pred):
                c = int(cls)  # integer class
                annotator.box_label(xyxy, f'{self._model.names[c]} {conf:.2f}', color=colors(c, True))
        img = annotator.result()
        return img

    def add_img(self, uid, img) -> bool:
        if uid in self._img_infos:
            warnings.warn("该uid已存在", UserWarning)
            return False
        if (len(self._img_infos) >= self._max_cache):
            uid_rm = self._img_uid_fifo.get()
            if not self._img_infos[uid_rm].is_used:
                warnings.warn(f'弹出uid={uid_rm}-未被使用', UserWarning)
            with self._img_infos[uid_rm].lock:
                self._img_infos.pop(uid_rm)
            print(f'YOLOv5缓存已满 弹出uid={uid_rm}')
        self._img_uid_fifo.put(uid)
        self._img_infos[uid] = YOLOV5Impl.ImgInfo()
        self._img_infos[uid].img = copy.deepcopy(img)

        # 处理图片
        img = letterbox(self._img_infos[uid].img, self._imgsz, stride=self._model.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        self._img_infos[uid].imgProcessed = copy.deepcopy(img)
        return True
