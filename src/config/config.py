from src.utils import singleton
import yaml
from typing import List, Dict

CONFIG_PATH = './.config.yml'

class WeightInfo:
    file: str
    labels: List[str]

@singleton
class Config:
    def __init__(self):
        # service
        self.service_name: str
        self.service_port: str
        self.service_tags: List[str] = list()
        self.weights_map: Dict[str, WeightInfo] = dict()

        #consul
        self.consul_ip: str
        self.consul_port: str
        
        self.load_config()

    def load_config(self):
        with open(CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)

        service_data = config_data.get('service', {})
        self.service_name = service_data.get('name', '')
        self.service_port = str(service_data.get('port', ''))
        self.service_tags = service_data.get('tags', [])
        weights = service_data.get('weights', [])
        for weight in weights:
            weight_info = WeightInfo()
            weight_info.file = weight['file']
            weight_info.labels = weight['labels']
            self.weights_map[weight_info.file] = weight_info
        
        consul_data = config_data.get('consul', {})
        self.consul_ip = consul_data.get('ip', '')
        self.consul_port = consul_data.get('port', '')
