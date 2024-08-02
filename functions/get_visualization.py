import cv2
import numpy as np
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from batch_face import RetinaFace
from tensorflow.keras.preprocessing.image import img_to_array

class VideoCamera(object):
    def __init__(self, path_video='', path_report='', path_save='', name_labels = '', conf=0.7):
        self.path_video = path_video
        self.df = pd.read_csv(path_report)
        self.prob = pd.DataFrame(self.df.drop(['frame'], axis=1)).values
        self.sort_pred = np.argsort(-self.prob)
        self.labels = name_labels
        self.path_save = path_save
        self.conf = conf
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cur_frame = 0
        self.video = None
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
        
    def draw_prob(self, emotion_yhat, best_n, startX, startY, endX, endY):
    
        label = '{}: {:.2f}%'.format(self.labels[best_n[0]], emotion_yhat[best_n[0]]*100)
        
        lw = max(round(sum(self.fr.shape) / 2 * 0.003), 2)
        
        text2_color = (255, 0, 255)
        p1, p2 = (startX, startY), (endX, endY)
        cv2.rectangle(self.fr, p1, p2, text2_color, thickness=lw, lineType=cv2.LINE_AA)
                
        tf = max(lw - 1, 1)
        fontScale = 2
        text_fond = (0,0,0)
        text_width_2, text_height_2 = cv2.getTextSize(label, self.font, lw / 3, tf)
        text_width_2 = text_width_2[0]+round(((p2[0]-p1[0])*10)/360)
        center_face = p1[0]+round((p2[0]-p1[0])/2)
        
        cv2.putText(self.fr, label, (center_face-round(text_width_2/2), p1[1] - round(((p2[0]-p1[0])*20)/360)), 
                    self.font, lw / 3, text_fond, thickness=tf, lineType=cv2.LINE_AA)
        
        cv2.putText(self.fr, label, (center_face-round(text_width_2/2), p1[1] - round(((p2[0]-p1[0])*20)/360)), 
                    self.font, lw / 3, text2_color, thickness=tf, lineType=cv2.LINE_AA)
            
    def get_video(self):
        self.video = cv2.VideoCapture(self.path_video)
        total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = np.round(self.video.get(cv2.CAP_PROP_FPS))
        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.path_save += '.mp4'
        vid_writer = cv2.VideoWriter(self.path_save, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        while True:
            _, self.fr = self.video.read()
            if self.fr is None: break
            faces = self.detector(self.fr, cv=False)
            for f_id, box in enumerate(faces):
                box, _, prob = box
                if prob > self.conf:
                    startX = int(box[0])
                    startY = int(box[1])
                    endX = int(box[2])
                    endY = int(box[3])
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    prob = self.prob[self.cur_frame]
                    best = self.sort_pred[self.cur_frame]
                    self.draw_prob(prob, best, startX, startY, endX, endY)
                        
            self.cur_frame += 1

            vid_writer.write(self.fr)
        vid_writer.release()