from flask import Flask, make_response
import cv2
from yolov5_src import YOLOV5Impl
builder = YOLOV5Impl.YOLOV5Builder()
builder._device = 'cuda:0'
impl = builder.build()
impl.load_model()

app = Flask(__name__)

@app.route('/get_img_labeled/<uid>')
def get_img_labeled(uid):
    uid = int(uid)
    img1 = impl.get_imglabeled_by_uid(uid)
    cv2.imwrite(f'{uid}.jpg', img1)
    image_data = open(f'{uid}.jpg', "rb").read()
    response = make_response(image_data)
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
