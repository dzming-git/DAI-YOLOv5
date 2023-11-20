from generated.protos.image_harmony import image_harmony_pb2, image_harmony_pb2_grpc
import grpc
import cv2
from typing import Dict, Tuple
import numpy as np

# TODO: 粗糙实现
class ImageHarmonyClient:
    def __init__(self, ip:str, port: str):
        options = [('grpc.max_receive_message_length', 1024 * 1024 * 1024)]
        self.conn = grpc.insecure_channel(f'{ip}:{port}', options=options)
        self.client = image_harmony_pb2_grpc.CommunicateStub(channel=self.conn)
        self.connect_id: int = 0
    
    def __del__(self):
        if -1 != self.connect_id:
            unregister_image_harmony_service_request = image_harmony_pb2.UnregisterImgTransServiceRequest()
            unregister_image_harmony_service_request.connectId = self.connect_id
            unregister_image_harmony_service_response = self.client.unregisterImgTransService(unregister_image_harmony_service_request)
            
            response = unregister_image_harmony_service_response.response
            print(f'{response.code}: {response.message}')

    
    def set_connect_id(self, connect_id: int):
        self.connect_id = connect_id
    
    def set_args(self, img_type: str, args: Dict[str, str]):
        register_image_harmony_service_request = image_harmony_pb2.RegisterImgTransServiceRequest()
        register_image_harmony_service_request.imgType = img_type
        for key, value in args.items():
            arg = register_image_harmony_service_request.args.add()
            arg.key = key
            arg.value = value
        register_image_harmony_service_response = self.client.registerImgTransService(register_image_harmony_service_request)
        self.connect_id = register_image_harmony_service_response.connectId
        response = register_image_harmony_service_response.response
        print(f'{response.code}: {response.message}')
    
    def get_img(self) -> Tuple[int, cv2.Mat]:
        get_img_request = image_harmony_pb2.GetImgRequest()
        get_img_request.connectId = self.connect_id
        # TODO: 这些参数暂时固定
        get_img_request.format = '.jpg'
        get_img_request.params.extend([cv2.IMWRITE_JPEG_QUALITY, 80])
        get_img_request.expectedW = 640
        get_img_request.expectedH = 640
        get_img_response = self.client.getImg(get_img_request)
        response = get_img_response.response
        if 200 != response.code:
            print(f'{response.code}: {response.message}')
            return -1, cv2.Mat()
        img_id = get_img_response.imgId
        buf = get_img_response.buf
        nparr = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img_id, img
