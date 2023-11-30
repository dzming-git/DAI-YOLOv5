from src.utils import singleton
from typing import Dict
from src.wrapper.yolov5_detector import YOLOv5Detector
from src.config.config import Config

class DetectorInfo:
    def __init__(self) -> None:
        self.weight_path: str
        self.detector: YOLOv5Detector = None
        self.cnt: int = 0

@singleton
class DetectorManager:
    def __init__(self):
        self.detector_info_map: Dict[str, DetectorInfo] = {}
        config = Config()
        # TODO 测试，试一下初始化时就创建好detector
        # 不加载model，资源消耗很小            
        yolov5_builder = YOLOv5Detector.YOLOv5Builder()
        for weight in config.weights_map:
            self.detector_info_map[weight] = DetectorInfo()
            yolov5_builder.weight_path = f'./weights/{weight}'
            self.detector_info_map[weight].detector = yolov5_builder.build()
            self.detector_info_map[weight].weight_path = f'./weights/{weight}'
            self.detector_info_map[weight].cnt = 0
