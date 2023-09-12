from flask import Flask, make_response
import cv2
from yolov5_src import YOLOV5Impl
builder = YOLOV5Impl.YOLOV5Builder()
builder._device = 'cuda:0'
impl = builder.build()
impl.load_model()

app = Flask(__name__)

@app.route('/get_img_labeled/<uid>.jpg')
def get_img_labeled(uid):
    uid = int(uid)
    img1 = impl.get_imglabeled_by_uid(uid)
    _, compressed_image = cv2.imencode('.jpg', img1, [cv2.IMWRITE_JPEG_QUALITY, 90])
    image_bytes = compressed_image.tobytes()
    response = make_response(image_bytes)
    response.headers['Content-Type'] = 'image/jpg'
    return response

def yolov5_test():
    img = cv2.imread('yolov5_src/data/images/bus.jpg')
    uid = 1
    impl.add_img(uid, img)
    impl.detect_by_uid(uid)

    img = cv2.imread('yolov5_src/data/images/zidane.jpg')
    uid = 2
    impl.add_img(uid, img)
    impl.detect_by_uid(uid)
    
if __name__ == '__main__':
    yolov5_test()
    app.run(host='127.0.0.1', port='8001')
