from typing import List

class ServiceCheck:
    def __init__(self):
        self.protocol: str = 'TCP'
        self.url: str = ''
        self.status: str = 'passing'
        self.interval_ms: int = 10000
        self.timeout_ms: int = 3000

class ServiceInfo:
    def __init__(self):
        self.service_id: str
        self.service_name: str
        self.service_ip: str
        self.service_port: str
        self.service_tags: List[str] = list()
        self.service_check: ServiceCheck = ServiceCheck()
