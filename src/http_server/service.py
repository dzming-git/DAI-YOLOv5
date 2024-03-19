import warnings
from src.wrapper.yolov5_detector import YOLOv5Detector
from flask import Flask, make_response, request, render_template
from werkzeug.serving import make_server, BaseWSGIServer
import cv2
import hashlib

class YOLOv5Service:
    # 单例模式+建造者模式
    class SingletonBuilder:
        __instance = None

        @staticmethod
        def get_instance() -> 'YOLOv5Service':
            if YOLOv5Service.SingletonBuilder.__instance is None:
                warnings.warn('yolov5 service unbuilt!')
            return YOLOv5Service.SingletonBuilder.__instance
        
        def __init__(self):
            self.host: str = "127.0.0.1"
            self.port: str = "5000"
            self.template_folder: str = 'templates'
        
        def build(self) -> None:
            YOLOv5Service.SingletonBuilder.__instance = YOLOv5Service(self)
    
    def __init__(self,builder: SingletonBuilder):
        self.__host = builder.host
        self.__port = builder.port
        self.__server = self.__creat_server(template_folder=builder.template_folder)
        self._detector: YOLOv5Detector = None

    def set_detector(self, detector: YOLOv5Detector) -> bool:
        is_ok = detector.load_model()
        if is_ok:
            self._detector = detector
        return is_ok
    
    def __creat_server(self, template_folder: str) -> BaseWSGIServer:
        app = Flask(__name__,
                    template_folder=template_folder)

        @app.route('/', methods=['GET', 'POST'])
        def index():
            '''
            简单demo，输入一个图像地址，显示检测结果
            '''
            image_bytes = None
            if request.method == 'POST':
                url = request.form['input_text']
                cap = cv2.VideoCapture(url)
                ret, img = cap.read()
                cap.release()
                if not ret:
                    return render_template('index.html')

                md5_hash = hashlib.md5()
                md5_hash.update(url.encode('utf-8'))
                image_id = int(md5_hash.hexdigest(), 16)

                self._detector.add_image(image_id, img)
                self._detector.detect_by_image_id(image_id)

                img1 = self._detector.get_labeled_image_by_image_id(image_id)
                _, compressed_image = cv2.imencode('.jpg', img1, [cv2.IMWRITE_JPEG_QUALITY, 90])
                image_bytes = compressed_image.tobytes()
                response = make_response(image_bytes)
                response.headers['Content-Type'] = 'image/jpg'
                return response
            return render_template('index.html')

        server = make_server(host=self.__host, port=int(self.__port), app=app, threaded=True)
        return server
    
    def start_server(self):
        self.__server.serve_forever()
