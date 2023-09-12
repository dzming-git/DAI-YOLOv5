import warnings
from yolov5_src import YOLOV5Impl
from flask import Flask, make_response, request, render_template
from werkzeug.serving import make_server, BaseWSGIServer
import cv2
import hashlib

class YOLOv5Service:
    def set_yolov5_impl(self, yolov5_impl:YOLOV5Impl) -> bool:
        pass
    
    def creat_server(self) -> BaseWSGIServer:
        pass
    
    def start_server(self):
        pass

class YOLOv5Service:
    # 单例模式+建造者模式
    class SingletonBuilder:
        _instance = None

        @staticmethod
        def get_instance() -> YOLOv5Service:
            if YOLOv5Service.SingletonBuilder._instance is None:
                warnings.warn('yolov5 service unbuilt!')
            return YOLOv5Service.SingletonBuilder._instance
        
        def __init__(self):
            self._host = "127.0.0.1"
            self._port = "8001"
                
        def set_host(self, host:str) -> None:
            self._host = host
        
        def get_host(self) ->str:
            return self._host

        def set_port(self, port:str) -> None:
            self._port = port
        
        def get_port(self) -> str:
            return self._port
        
        def build(self) -> None:
            YOLOv5Service.SingletonBuilder._instance = YOLOv5Service(self)
    
    def __init__(self,builder: SingletonBuilder):
        self._host = builder.get_host()
        self._port = builder.get_port()
        self._server = self._creat_server()
        self._impl = None

    def set_yolov5_impl(self, yolov5_impl:YOLOV5Impl) -> bool:
        is_ok = True
        if yolov5_impl._model is None:
            is_ok = yolov5_impl.load_model()
        if is_ok:
            self._impl = yolov5_impl
        return is_ok
    
    def _creat_server(self) -> BaseWSGIServer:
        app = Flask(__name__)

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
                uid = int(md5_hash.hexdigest(), 16)

                self._impl.add_img(uid, img)
                self._impl.detect_by_uid(uid)

                img1 = self._impl.get_imglabeled_by_uid(uid)
                _, compressed_image = cv2.imencode('.jpg', img1, [cv2.IMWRITE_JPEG_QUALITY, 90])
                image_bytes = compressed_image.tobytes()
                response = make_response(image_bytes)
                response.headers['Content-Type'] = 'image/jpg'
                return response
            return render_template('index.html')

        server = make_server(host=self._host, port=int(self._port), app=app)
        return server
    
    def start_server(self):
        self._server.serve_forever()
