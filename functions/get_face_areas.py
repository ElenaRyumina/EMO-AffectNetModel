import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_vggface import utils
from tensorflow.keras.preprocessing.image import img_to_array
from batch_face import RetinaFace

class VideoCamera(object):
    def __init__(self, path_video='', conf=0.7):
        self.path_video = path_video
        self.conf = conf
        self.cur_frame = 0
        self.video = None
        self.dict_face_area = {}
        self.detector = RetinaFace(gpu_id=0)

    def __del__(self):
        self.video.release()
        
    def preprocess_image(self, cur_fr):
        cur_fr = utils.preprocess_input(cur_fr, version=2)
        return cur_fr
        
    def channel_frame_normalization(self, cur_fr):
        cur_fr = cv2.cvtColor(cur_fr, cv2.COLOR_BGR2RGB)
        cur_fr = cv2.resize(cur_fr, (224,224), interpolation=cv2.INTER_AREA)
        cur_fr = img_to_array(cur_fr)
        cur_fr = self.preprocess_image(cur_fr)
        return cur_fr
            
    def get_frame(self):
        self.video = cv2.VideoCapture(self.path_video)
        total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = np.round(self.video.get(cv2.CAP_PROP_FPS))
        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        need_frames = list(range(1, total_frame+1, round(5*fps/25)))
        print('Name video: ', os.path.basename(self.path_video))
        print('Number total of frames: ', total_frame)
        print('FPS: ', fps)
        print('Video duration: {} s'.format(np.round(total_frame/fps, 2)))
        print('Frame width:', w)
        print('Frame height:', h)
        while True:
            _, self.fr = self.video.read()
            if self.fr is None: break
            self.cur_frame += 1
            name_img = str(self.cur_frame).zfill(6)
            faces = self.detector(self.fr, cv=False)
            for j in need_frames:
                if self.cur_frame == j:
                    for f_id, box in enumerate(faces):
                        box, _, prob = box
                        if prob > self.conf:
                            startX = int(box[0])
                            startY = int(box[1])
                            endX = int(box[2])
                            endY = int(box[3])
                            (startX, startY) = (max(0, startX), max(0, startY))
                            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                            cur_fr = self.fr[startY: endY, startX: endX]
                            self.dict_face_area[name_img] = self.channel_frame_normalization(cur_fr)
        del self.detector          
        return self.dict_face_area, total_frame