class GRPCServer:
    def __init__(self, builder):
        import grpc
        from concurrent import futures
        address = f'{builder.ip}:{builder.port}'
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=builder.max_workers))
        self.server.add_insecure_port(address)
        print(f'grpc server address: {address}')
    
    def start(self):
        self.server.start()
