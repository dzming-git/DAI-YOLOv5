import numpy as np
import torch
from yolov5_src.models.experimental import attempt_load
from yolov5_src.utils.general import non_max_suppression, scale_boxes, check_img_size
from yolov5_src.utils.plots import Annotator, colors
from yolov5_src.utils.augmentations import letterbox
from yolov5_src.utils.torch_utils import select_device
from yolov5_src.models.common import DetectMultiBackend



class YOLOV5Impl:
    class YOLOV5Builder:
        def __init__(self):
            self._weights = 'yolov5_src/yolov5s.pt'
            self._device = torch.device('cpu')
            self._imgsz = [640] * 2
            self._conf_thres = 0.5
            self._iou_thres = 0.45
            self._max_det = 1000
            self._agnostic_nms = False
            self._augment = False
            self._half = False
            self._dnn = False
            self._data = 'yolov5_src/data/coco128.yaml'

            self._print_result = False
        
        def build(self):
            return YOLOV5Impl(self)

    def __init__(self, builder:YOLOV5Builder):
        self._weights = builder._weights
        self._device = builder._device
        self._imgsz = builder._imgsz
        self._conf_thres = builder._conf_thres
        self._iou_thres = builder._iou_thres
        self._max_det = builder._max_det
        self._agnostic_nms = builder._agnostic_nms
        self._augment = builder._augment
        self._half = builder._half & (self._device.type != 'cpu')
        self._dnn = builder._dnn
        self._data = builder._data
        self._model, self._stride, self._names = self._loadModel()
        self._pred = np.zeros((0, 0, 0))
        self._img = None
        self._imgProcessed = None

    def _loadModel(self):
        model = DetectMultiBackend(self._weights, device=self._device, dnn=self._dnn, data=self._data, fp16=self._half)
        stride, names, pt = model.stride, model.names, model.pt
        self._imgsz = check_img_size(self._imgsz, s=stride)  # check image size
        return model, stride, names

    def _imgProcess(self):
        img = letterbox(self._img, self._imgsz, stride=self._stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        self._imgProcessed = img

    @torch.no_grad()
    def _det(self):
        pred = self._model(self._imgProcessed)[0]
        self._pred = non_max_suppression(prediction=pred,
                                          conf_thres=self._conf_thres,
                                          iou_thres=self._iou_thres,
                                          max_det=self._max_det)[0]
        if len(self._pred):
            self._pred[:, :4] = scale_boxes(self._imgProcessed.shape[2:], self._pred[:, :4], self._img.shape).round()

    def getResult(self):
        result = []
        if len(self._pred):
            for *xyxy, conf, cls in reversed(self._pred):
                c = int(cls)  # integer class
                label = f'{self._names[c]}'
                imgRows, imgCols, _ = self._img.shape
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

    def getImgLabeled(self):
        annotator = Annotator(self._img, line_width=3, example=str(self._names))
        if len(self._pred):
            for *xyxy, conf, cls in reversed(self._pred):
                c = int(cls)  # integer class
                label = f'{self._names[c]}'
                annotator.box_label(xyxy, f'{self._names[c]} {conf:.2f}', color=colors(c, True))
        img = annotator.result()
        return img

    def addImg(self, img):
        self._img = img
        self._imgProcess()
        self._det()
