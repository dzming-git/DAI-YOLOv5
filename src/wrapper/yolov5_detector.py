SUBMODULE_DIR = '/workspace/yolov5'

import numpy as np
import torch
import cv2
from src.utils import class_temporary_change_dir, temporary_change_dir
with temporary_change_dir(SUBMODULE_DIR):
    from yolov5.utils.general import non_max_suppression, scale_boxes, check_img_size
    from yolov5.utils.plots import Annotator, colors
    from yolov5.utils.augmentations import letterbox
from typing import Dict, List, Tuple, Optional
import queue
import copy
import threading
import traceback
from src.model_manager.model_manager import ModelManager, ModelInfo
import warnings
warnings.filterwarnings('always')

# Step 说明
DETECT_IMAGE_COMPLETE = 0
DETECT_IMAGE_START = 1
ADD_IMAGE_COMPLETE = 2
ADD_IMAGE_START = 3
ADD_UID_COMPLETE = 4

model_manager = ModelManager()

@class_temporary_change_dir(SUBMODULE_DIR)
class YOLOv5Detector:
    class YOLOv5Builder:
        def __init__(self) -> None:
            self.weight: str = 'yolov5s.pt'
            self.device_str: str = 'cpu'
            self.imgsz: List[int] = [640, 640]
            self.conf_thres: float = 0.5
            self.iou_thres: float = 0.45
            self.max_det: int = 1000
            self.half: bool = False
            self.dnn: bool = False

            # 是否将缓存图像，默认不缓存
            self.save_image_to_cache: bool = False

            # 最大缓存100个图像结果
            self.max_cache: int = 100

            # get_result_by_image_id 是否打印结果
            self.print_result: bool = False
        
        def build(self) -> 'YOLOv5Detector':
            if not torch.cuda.is_available() and self.device_str != 'cpu':
                warnings.warn("CUDA is not available", UserWarning)
                self.device_str = 'cpu'
            return YOLOv5Detector(self)

    class ImageInfo:
        def __init__(self) -> None:
            self.image_id: int = 0
            self.image: Optional[np.ndarray] = None
            self.image_shape: np.ndarray = np.zeros((0, 0, 0))
            self.processed_image: Optional[torch.Tensor] = None
            self.pred: np.ndarray = np.zeros((0, 0, 0))
            self.is_used: bool = False
            self.step: int = ADD_UID_COMPLETE
            self.lock: threading.Lock = threading.Lock()

    def __init__(self, builder: 'YOLOv5Builder') -> None:
        self.__weight: str = builder.weight
        self.__device: torch.device = torch.device(builder.device_str)
        self.__imagesz: List[int] = builder.imgsz
        self.__conf_thres: float = builder.conf_thres
        self.__iou_thres: float = builder.iou_thres
        self.__max_det: int = builder.max_det
        self.__half: bool = builder.half
        self.__dnn: bool = builder.dnn
        self.__max_cache: int = builder.max_cache
        self.__print_result: bool = builder.print_result

        self.__model_info: ModelInfo = model_manager.get_model_info(self.__weight, self.__device, self.__dnn, self.__half)

        self.__save_image_to_cache: bool = builder.save_image_to_cache
        self.__image_infos: Dict[int, YOLOv5Detector.ImageInfo] = {}
        self.__image_image_id_fifo: queue.Queue[int] = queue.Queue(maxsize=self.__max_cache)
        
        self.latest_detection_completed_image_id: int = 0
        self.latest_add_image_id: int = 0

    def load_model(self) -> None:
        self.__model_info.start_using()
        self.__imagesz = check_img_size(self.__imagesz, s=self.__model_info.model.stride)  # check image size

    def check_image_id_exist(self, image_id: int) -> bool:
        return image_id in self.__image_infos

    def get_status(self, image_id: int) -> int:
        if not self.check_image_id_exist(image_id):
            warnings.warn("该image id不存在", UserWarning)
            return -1
        return self.__image_infos[image_id].step

    def add_image_id(self, image_id: int) -> bool:
        if self.check_image_id_exist(image_id):
            warnings.warn("该image id已存在", UserWarning)
            return False
        # 清理溢出
        if (len(self.__image_infos) >= self.__max_cache):
            image_id_rm = self.__image_image_id_fifo.get()
            if not self.__image_infos[image_id_rm].is_used:
                warnings.warn(f'弹出image id={image_id_rm}-未被使用', UserWarning)
            with self.__image_infos[image_id_rm].lock:
                self.__image_infos.pop(image_id_rm)
        self.__image_image_id_fifo.put(image_id)
        self.__image_infos[image_id] = YOLOv5Detector.ImageInfo()
        self.__image_infos[image_id].step = ADD_UID_COMPLETE
        return True

    @torch.no_grad()
    def detect_by_image_id(self, image_id: int) -> bool:
        if not self.check_image_id_exist(image_id):
            warnings.warn("该image id不存在", UserWarning)
            return False
        if self.__image_infos[image_id].step != ADD_IMAGE_COMPLETE:
            warnings.warn("图片未添加完成", UserWarning)
            return False
        self.__image_infos[image_id].step = DETECT_IMAGE_START
        with self.__image_infos[image_id].lock:
            pred = self.__model_info.model(self.__image_infos[image_id].processed_image)[0]
        pred = non_max_suppression(prediction=pred,
                                        conf_thres=self.__conf_thres,
                                        iou_thres=self.__iou_thres,
                                        max_det=self.__max_det)[0]
        with self.__image_infos[image_id].lock:
            if len(pred):
                pred[:, :4] = scale_boxes(self.__image_infos[image_id].processed_image.shape[2:], pred[:, :4], self.__image_infos[image_id].image_shape).round()
            self.__image_infos[image_id].pred = pred
        
        del self.__image_infos[image_id].processed_image
        self.__image_infos[image_id].processed_image = None
        self.latest_detection_completed_image_id = image_id
        self.__image_infos[image_id].step = DETECT_IMAGE_COMPLETE

    def get_result_by_image_id(self, image_id: int):
        result = []
        if not self.check_image_id_exist(image_id):
            warnings.warn("该image id不存在", UserWarning)
            return None
        if self.__image_infos[image_id].step != DETECT_IMAGE_COMPLETE:
            warnings.warn("图片未添加完成", UserWarning)
            return False
        with self.__image_infos[image_id].lock:
            self.__image_infos[image_id].is_used = True
            if len(self.__image_infos[image_id].pred):
                for *xyxy, conf, cls in reversed(self.__image_infos[image_id].pred):
                    c = int(cls)  # integer class
                    # label = f'{self.__model_info.model.names[c]}'
                    h, w, _ = self.__image_infos[image_id].image_shape
                    result.append([
                        xyxy[0].item() / w,
                        xyxy[1].item() / h,
                        xyxy[2].item() / w,
                        xyxy[3].item() / h,
                        c,
                        float(conf.cpu().numpy())
                    ])
                    if self.__print_result:
                        print('YoloImgInterface', result[-1])
        return result

    def get_labeled_image_by_image_id(self, image_id: int):
        if not self.__save_image_to_cache:
            warnings.warn("save_image_to_cache is False", UserWarning)
            return np.ndarray()

        with self.__image_infos[image_id].lock:
            annotator = Annotator(self.__image_infos[image_id].image, line_width=3, example=str(self.__model_info.model.names))
        self.__image_infos[image_id].is_used = True
        if len(self.__image_infos[image_id].pred):
            for *xyxy, conf, cls in reversed(self.__image_infos[image_id].pred):
                c = int(cls)  # integer class
                annotator.box_label(xyxy, f'{self.__model_info.model.names[c]} {conf:.2f}', color=colors(c, True))
        image = annotator.result()
        return image

    def add_image(self, image_id: int, image: np.ndarray) -> bool:
        if not self.check_image_id_exist(image_id):
            self.add_image_id(image_id)
        elif self.__image_infos[image_id].step != ADD_UID_COMPLETE:
            print(self.__image_infos[image_id].step)
            warnings.warn("重复添加", UserWarning)
            return False
        self.__image_infos[image_id].step = ADD_IMAGE_START
        self.__image_infos[image_id].image_shape = image.shape
        if self.__save_image_to_cache:
            self.__image_infos[image_id].image = copy.deepcopy(image)
        # 处理图片
        image = letterbox(image, self.__imagesz, stride=self.__model_info.model.stride)[0]
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.__device)
        image = image.half() if self.__half else image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim
        self.__image_infos[image_id].processed_image = image
        self.latest_add_image_id = image_id
        self.__image_infos[image_id].step = ADD_IMAGE_COMPLETE
        return True

    def get_letterbox_size(self, width: int, height: int):
        # letterbox改写，只获取图像尺寸，没有图像处理
        shape = (height, width)
        new_shape = self.__imagesz
        auto = True
        scaleFill = False
        scaleup = True
        stride = self.__model_info.model.stride
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        # if shape[::-1] != new_unpad:  # resize
        #     im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return new_unpad[0], new_unpad[1], top, bottom, left, right