#Cấu hình thư viện sử dụng trong dự án:
#thư viện điều khiển hệ điều hành.
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Cấu hình tensorflow logging
#thư viện điều khiển đường dẫn đến các tập tin dự án
import pathlib
#thư viện tensorflow
import tensorflow as tf
#thư viện OpenCV
import cv2
#thư viện hỗ trợ các cú pháp bổ sung lập trình dự án
import argparse
#thư viện hỗ trợ giao diện xử lý đa luồng
from threading import Thread
#thư viện hỗ trợ phát mp3
from pydub import AudioSegment
from pydub.playback import play
temp = ""
#phat am thanh bao bat dau khoi dong :
song = AudioSegment.from_wav("start")
play(song)
#in thông báo lên màn hình
print('Dang tai du lieu. Vui long cho...')
tf.get_logger().setLevel('ERROR')           # thông báo nếu xuất hiện lỗi
#chương trình hiển thị video lên màn hình
class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        #Khởi tạo camera:
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Kiểm tra khung hình đầu tiên:
        (self.grabbed, self.frame) = self.stream.read()

    # Biến xác nhận có camera hay không:
        self.stopped = False

    def start(self):
    # bắt đầu phần luồng để đọc video từ camera
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Chương trình chính:
        while True:
            # Nếu camera bị rút ra -> dừng chương trình:
            if self.stopped:
                # Đóng camera
                self.stream.release()
                return

            # Nếu camera online -> chạy chương trình tiếp
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Lọc khung hình gần nhất
        return self.frame

    def stop(self):
    # biến kiểm tra camera
        self.stopped = True
        
#tải cơ sở dữ liệu
parser = argparse.ArgumentParser()
#tải models
parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='od-models/my_mobilenet_model')
#tải nhãn
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='models/research/object_detection/data/mscoco_label_map.pbtxt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
                    
args = parser.parse_args()


# Khai báo thư mục chứa models:
PATH_TO_MODEL_DIR = args.model

# Khai báo thư mục chứa các nhãn:
PATH_TO_LABELS = args.labels

# Biến lưu dữ lệu phân luồng:
MIN_CONF_THRESH = float(args.threshold)

# Tai du lieu
# ~~~~~~~~~~~~~~
#thư viện định giờ:
import time
#load thư viện object_detection:
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
#load model :
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
#khởi tạo hệ thống:
print('Dang tai du lieu ...', end='')
start_time = time.time()

# Tai cac model da luu
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
#đếm thời gian khởi động
end_time = time.time()
elapsed_time = end_time - start_time
#in lên màn hình thời gian khởi động sp:
print('Hoan tat! Mat {} giay'.format(elapsed_time))
#phat am thanh thong bao hoat tat
song = AudioSegment.from_wav("finish")
play(song)
# Tai du lieu chua cac nhan dat ten
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
#thêm các thư viện phụ:
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # tắt cảnh báo của matplotlib
#thông báo:
print('Chay phan mem tren Camera. Vui long cho...')
#khởi tạo camera ở độ phân giải 640x480, 30 khung hình /1 s
videostream = VideoStream(resolution=(640,480),framerate=30).start()
#chương trình nhận diện vật thể:
while True:

    # lấy khung hình:
    frame = videostream.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    imH, imW, _ = frame.shape

    # Chuyển đổi khung hình bằng lệnh `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(frame)
    # Chuyển đổi định dạng ảnh `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # Tiến hành so snahs.
    # chuyển ảnh sang numpy arrays.
    # đếm số lượng vật thể trong ảnh:
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes khai báo là ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    
    # So sánh kết quả
    
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    count = 0
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
            #tăng điểm tin cậy
            count += 1
            #vẽ khung hình bao quanh vật thể:
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            # Viết tên vật thể lên góc khung hình
            #so sánh tên , phát âm thanh:
            object_name = category_index[int(classes[i])]['name'] # Look up object name from "labels" array using class index
            ten = "%s" % (object_name)
            dotincay = int(scores[i]*100)
            label = '%s: %d%%' % (object_name, dotincay) # Example: 'person: 72%'
            if ten != temp: #neu ten vat the khong duoc nhac lai
                print("Label là: {}".format(ten))
                if dotincay >= 60: #phat am thanh khi do tin cay dat 60% tro len
                    play(AudioSegment.from_wav(ten))
            temp = ten #gan ten vat the vao bien tam
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
    #hiển thị cửa sổ giao diện
    cv2.putText (frame,'Phat hien vat the : ' + str(count),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),2,cv2.LINE_AA)
    cv2.imshow('Kinh mat cho nguoi khiem thi', frame)
    #nếu phím q được bấm:
    if cv2.waitKey(1) == ord('q'):
        break
#đóng chương trình
cv2.destroyAllWindows()
print("Thoat chuong trinh....")
videostream.stop()
