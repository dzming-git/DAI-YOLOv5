from src.utils import singleton
import yaml
from typing import List, Dict
import os

CONFIG_PATH = './.config.yml'

class WeightInfo:
    file: str
    labels: List[str]

@singleton
class Config:
    def __init__(self):
        # service
        self.service_name: str = ''
        self.service_port: str = ''
        self.service_tags: List[str] = list()
        self.weights_map: Dict[str, WeightInfo] = dict()
        self.weights_data_map: Dict[str, str] = dict()
        self.weights_folder: str = ''

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
        weight_info_folder = service_data.get('weight_info_folder', '')
        weight_info_paths = []
        
        if weight_info_folder:
            for root, dirs, files in os.walk(weight_info_folder):
                file_name: str
                for file_name in files:
                    if file_name.endswith('.yml') or file_name.endswith('yaml'):
                        file_path = os.path.join(root, file_name)
                        weight_info_paths.append(file_path)
        for weight_info_path in weight_info_paths:
            with open(weight_info_path, 'r') as f:
                filename = os.path.basename(weight_info_path)
                weight_info_data = yaml.safe_load(f)
                weight_info = WeightInfo()
                weight_info.file = os.path.splitext(filename)[0]
                data_path = weight_info_data['data_path']
                self.weights_data_map[weight_info.file] = data_path
                with open(data_path, 'r') as f_data:
                    data_data = yaml.safe_load(f_data)
                    weight_info.labels = list(data_data['names'].values())
                self.weights_map[weight_info.file] = weight_info
        
        self.weights_folder = service_data.get('weights_folder', '')
        
        consul_data = config_data.get('consul', {})
        self.consul_ip = consul_data.get('ip', '')
        self.consul_port = consul_data.get('port', '')
