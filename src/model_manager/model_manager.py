from src.utils import singleton
from typing import Dict
from src.config.config import Config
from yolov5.models.common import DetectMultiBackend
import threading
import torch
import traceback

# state说明
LOADING_COMPLETED = 0
LOADING = 1
NOT_LOADED = 2

config = Config()

class ModelInfo:
    def __init__(self, weight: str, device: torch.device, dnn: bool, half: bool) -> None:
        self.__weight: str = weight
        self.__data = config.weights_data_map[weight]
        self.__device: torch.device = device
        self.__dnn: bool = dnn
        self.__half: bool = half
        
        self.model: DetectMultiBackend = None
        self.__model_load_lock = threading.Lock()
        self.__user_cnt: int = 0
        self.__user_cnt_lock = threading.Lock()
        
        self.state = NOT_LOADED

    def get_state(self) -> int:
        return self.state

    def load(self) -> None:
        self.state = LOADING
        try:
            if self.model is None:
                with self.__model_load_lock:
                    if self.model is None:
                        weight_path = f'{config.weights_folder}/{self.__weight}'
                        self.model = DetectMultiBackend(
                            weights=weight_path, 
                            device=self.__device, 
                            dnn=self.__dnn, 
                            data=self.__data, 
                            fp16=self.__half
                        )
        except Exception as e:
            self.state = NOT_LOADED
            raise e from e
        self.state = LOADING_COMPLETED
    
    def start_using(self) -> None:
        with self.__user_cnt_lock:
            self.__user_cnt += 1
            if 1 == self.__user_cnt:
                self.load()
    
    def stop_using(self) -> None:
        with self.__user_cnt_lock:
            self.__user_cnt += 1
            if 0 == self.__user_cnt and self.model is not None:
                del self.model
                self.model = None

@singleton
class ModelManager:
    def __init__(self) -> None:
        self.model_info_map: Dict[int, ModelInfo] = {}

    def get_model_info(self, weight: str, device: torch.device, dnn: bool, half: bool) -> ModelInfo:
        values_tuple = (weight, device, dnn, half)
        hash_value = hash(values_tuple)
        if hash_value not in self.model_info_map:
            # model不存在，创建
            self.model_info_map[hash_value] = ModelInfo(weight, device, dnn, half)
        return self.model_info_map[hash_value]
