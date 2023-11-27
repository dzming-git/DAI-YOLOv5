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
        self.connect_id: int = 0
    
    def __del__(self):
        if 0 != self.connect_id:
            unregister_image_harmony_service_request = image_harmony_pb2.UnregisterImageTransServiceRequest()
            unregister_image_harmony_service_request.connectId = self.connect_id
            unregister_image_harmony_service_response = self.client.unregisterImageTransService(unregister_image_harmony_service_request)
            
            response = unregister_image_harmony_service_response.response
            print(f'{response.code}: {response.message}')

    
    def set_connect_id(self, connect_id: int):
        self.connect_id = connect_id
    
    def set_args(self, image_type: str, args: Dict[str, str]):
        register_image_harmony_service_request = image_harmony_pb2.RegisterImageTransServiceRequest()
        register_image_harmony_service_request.imageType = image_type
        for key, value in args.items():
            arg = register_image_harmony_service_request.args.add()
            arg.key = key
            arg.value = value
        register_image_harmony_service_response = self.client.registerImageTransService(register_image_harmony_service_request)
        self.connect_id = register_image_harmony_service_response.connectId
        response = register_image_harmony_service_response.response
        print(f'{response.code}: {response.message}')
    
    def get_image_by_image_id(self, image_id: int) -> Tuple[int, cv2.Mat]:
        get_image_by_image_id_request = image_harmony_pb2.GetImageByImageIdRequest()
        get_image_by_image_id_request.connectId = self.connect_id
        get_image_by_image_id_request.imageId = image_id
        # TODO: 这些参数暂时固定
        get_image_by_image_id_request.format = '.jpg'
        get_image_by_image_id_request.params.extend([cv2.IMWRITE_JPEG_QUALITY, 80])
        get_image_by_image_id_request.expectedW = 640
        get_image_by_image_id_request.expectedH = 640
        get_image_by_image_id_response = self.client.getImage(get_image_by_image_id_request)
        response = get_image_by_image_id_response.response
        if 200 != response.code:
            print(f'{response.code}: {response.message}')
            return 0, np.empty((0, 0, 0))
        image_id = get_image_by_image_id_response.imageId
        buf = get_image_by_image_id_response.buf
        nparr = np.frombuffer(buf, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image_id, image

    def get_latest_image(self) -> Tuple[int, cv2.Mat]:
        get_next_image_by_image_id_request = image_harmony_pb2.GetNextImageByImageIdRequest()
        get_next_image_by_image_id_request.connectId = self.connect_id
        get_next_image_by_image_id_request.imageId = 0
        # TODO: 这些参数暂时固定
        get_next_image_by_image_id_request.format = '.jpg'
        get_next_image_by_image_id_request.params.extend([cv2.IMWRITE_JPEG_QUALITY, 80])
        get_next_image_by_image_id_request.expectedW = 640
        get_next_image_by_image_id_request.expectedH = 640
        get_image_by_image_id_response = self.client.getNextImageByImageId(get_next_image_by_image_id_request)
        response = get_image_by_image_id_response.response
        if 200 != response.code:
            print(f'{response.code}: {response.message}')
            return 0, np.empty((0, 0, 0))
        image_id = get_image_by_image_id_response.imageId
        buf = get_image_by_image_id_response.buf
        nparr = np.frombuffer(buf, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image_id, image