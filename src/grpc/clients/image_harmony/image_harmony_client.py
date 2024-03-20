from generated.protos.image_harmony import image_harmony_pb2, image_harmony_pb2_grpc
import grpc
import cv2
from typing import Dict, Tuple
import numpy as np

# TODO 粗糙实现
class ImageHarmonyClient:
    def __init__(self, ip:str, port: str):
        options = [('grpc.max_receive_message_length', 1024 * 1024 * 1024)]
        self.conn = grpc.insecure_channel(f'{ip}:{port}', options=options)
        self.client = image_harmony_pb2_grpc.CommunicateStub(channel=self.conn)
        self.connection_id: int = 0
    
    def connect_image_loader(self, loader_args_hash: int):
        request = image_harmony_pb2.ConnectImageLoaderRequest()
        request.loaderArgsHash = loader_args_hash
        response = self.client.connectImageLoader(request)
        self.connection_id = response.connectionId
        print(f'{response.response.code}: {response.response.message}')

    def disconnect_image_loader(self):
        if 0 != self.connection_id:
            request = image_harmony_pb2.DisconnectImageLoaderRequest()
            request.connectionId = self.connection_id
            response = self.client.disconnectImageLoader(request)
            print(f'{response.response.code}: {response.response.message}')
    
    def set_connect_id(self, connect_id: int):
        self.connection_id = connect_id
    
    def set_args(self, image_type: str, args: Dict[str, str]):
        register_image_harmony_service_request = image_harmony_pb2.ConnectImageLoaderRequest()
        register_image_harmony_service_request.imageType = image_type
        for key, value in args.items():
            arg = register_image_harmony_service_request.args.add()
            arg.key = key
            arg.value = value
        register_image_harmony_service_response = self.client.registerImageLoader(register_image_harmony_service_request)
        self.connection_id = register_image_harmony_service_response.connectionId
        response = register_image_harmony_service_response.response
        print(f'{response.code}: {response.message}')
    
    def get_image_by_image_id(self, image_id: int, width: int, height: int) -> Tuple[int, np.ndarray]:
        get_image_by_image_id_request = image_harmony_pb2.GetImageByImageIdRequest()
        get_image_by_image_id_request.connectionId = self.connection_id
        get_image_by_image_id_request.imageRequest.imageId = image_id
        # TODO: 这些参数暂时固定
        get_image_by_image_id_request.imageRequest.format = '.jpg'
        get_image_by_image_id_request.imageRequest.params.extend([cv2.IMWRITE_JPEG_QUALITY, 80])
        get_image_by_image_id_request.imageRequest.expectedW = width
        get_image_by_image_id_request.imageRequest.expectedH = height
        get_image_by_image_id_response = self.client.getImageByImageId(get_image_by_image_id_request)
        response = get_image_by_image_id_response.response
        if 200 != response.code:
            print(f'{response.code}: {response.message}')
            return 0, np.empty((0), dtype=np.uint8)
        image_id = get_image_by_image_id_response.imageResponse.imageId
        buffer = get_image_by_image_id_response.imageResponse.buffer
        nparr = np.frombuffer(buffer, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image_id, image
    
    def get_image_size_by_image_id(self, image_id: int) -> Tuple[int, int]:
        get_image_by_image_id_request = image_harmony_pb2.GetImageByImageIdRequest()
        get_image_by_image_id_request.connectionId = self.connection_id
        get_image_by_image_id_request.imageRequest.imageId = image_id
        get_image_by_image_id_request.imageRequest.noImageBuffer = True
        get_image_by_image_id_response = self.client.getImageByImageId(get_image_by_image_id_request)
        response = get_image_by_image_id_response.response
        if 200 != response.code:
            print(f'{response.code}: {response.message}')
            return 0, np.empty((0), dtype=np.uint8)
        image_id = get_image_by_image_id_response.imageResponse.imageId
        width = get_image_by_image_id_response.imageResponse.width
        height = get_image_by_image_id_response.imageResponse.height
        return width, height

    def get_latest_image(self, width: int, height: int) -> Tuple[int, np.ndarray]:
        get_next_image_by_image_id_request = image_harmony_pb2.GetNextImageByImageIdRequest()
        get_next_image_by_image_id_request.connectionId = self.connection_id
        get_next_image_by_image_id_request.imageRequest.imageId = 0
        # TODO: 这些参数暂时固定
        get_next_image_by_image_id_request.imageRequest.format = '.jpg'
        get_next_image_by_image_id_request.imageRequest.params.extend([cv2.IMWRITE_JPEG_QUALITY, 80])
        get_next_image_by_image_id_request.imageRequest.expectedW = width
        get_next_image_by_image_id_request.imageRequest.expectedH = height
        get_image_by_image_id_response = self.client.getNextImageByImageId(get_next_image_by_image_id_request)
        response = get_image_by_image_id_response.response
        if 200 != response.code:
            print(f'{response.code}: {response.message}')
            return 0, np.empty((0), dtype=np.uint8)
        image_id = get_image_by_image_id_response.imageResponse.imageId
        buffer = get_image_by_image_id_response.imageResponse.buffer
        nparr = np.frombuffer(buffer, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image_id, image