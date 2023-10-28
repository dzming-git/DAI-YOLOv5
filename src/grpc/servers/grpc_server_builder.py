from src.grpc.servers.grpc_server import GRPCServer

class GRPCServerBuilder:
    def __init__(self):
        self.ip: str = '0.0.0.0'
        self.port: str = '5000'
        self.max_workers: int = 10
    
    def build(self) -> GRPCServer:
        gRPC_server = GRPCServer(self)
        return gRPC_server
