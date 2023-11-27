from src.utils import singleton
from typing import Dict
from src.wrapper.yolov5_detector import YOLOv5Detector

class DetectorInfo:
    def __init__(self) -> None:
        self.weight_path: str
        self.detector: YOLOv5Detector = None
        self.cnt: int = 0

@singleton
class DetectorManager:
    def __init__(self):
        self.detector_info_map: Dict[str, DetectorInfo] = {}

