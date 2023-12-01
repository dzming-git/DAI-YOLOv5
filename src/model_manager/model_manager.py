from src.utils import singleton
from typing import Dict
from src.config.config import Config
from yolov5_src.models.common import DetectMultiBackend
import threading
import warnings
import torch
warnings.filterwarnings('always')

# state说明
LOADING_COMPLETED = 0
LOADING = 1
NOT_LOADED = 2

config = Config()

class ModelInfo:
    def __init__(self, weight: str, device: torch.device, dnn: bool, half: bool) -> None:
        self._weight: str = weight
        self._data = config.weights_data_map[weight]
        self._device: torch.device = device
        self._dnn: bool = dnn
        self._half: bool = half
        
        self._model: DetectMultiBackend = None
        self._model_load_lock = threading.Lock()
        self._user_cnt: int = 0
        self._user_cnt_lock = threading.Lock()
        
        self.state = NOT_LOADED

    def get_state(self) -> int:
        return self.state

    def load(self) -> bool:
        self.state = LOADING
        if self._model is None:
            with self._model_load_lock:
                if self._model is None:
                    try:
                        weight_path = f'./weights/{self._weight}'
                        self._model = DetectMultiBackend(weights=weight_path, device=self._device, dnn=self._dnn, data=self._data, fp16=self._half)
                    except Exception as e:
                        warnings.warn(e, UserWarning)
                        self.state = NOT_LOADED
                        return False
        self.state = LOADING_COMPLETED
        return True
    
    def start_using(self) -> bool:
        with self._user_cnt_lock:
            self._user_cnt += 1
            if 1 == self._user_cnt:
                if not self.load():
                    return False
        return True
    
    def stop_using(self):
        with self._user_cnt_lock:
            self._user_cnt += 1
            if 0 == self._user_cnt and self._model is not None:
                del self._model
                self._model = None

@singleton
class ModelManager:
    def __init__(self):
        self.model_info_map: Dict[int, ModelInfo] = {}

    def get_model_info(self, weight: str, device: torch.device, dnn: bool, half: bool) -> ModelInfo:
        values_tuple = (weight, device, dnn, half)
        hash_value = hash(values_tuple)
        if hash_value not in self.model_info_map:
            # model不存在，创建
            self.model_info_map[hash_value] = ModelInfo(weight, device, dnn, half)
        return self.model_info_map[hash_value]
