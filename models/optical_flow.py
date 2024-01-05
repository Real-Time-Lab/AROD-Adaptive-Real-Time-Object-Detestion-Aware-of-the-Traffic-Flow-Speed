import torch
import cv2 as cv
import numpy as np
from mlp import load_model
import time
import os
import re
import pickle

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=50,
                      qualityLevel=0.1,
                      minDistance=2,
                      blockSize=3)
# Parameters for lucas kanade optical flow
# maxLevel
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))



class optical_flow():  ## optical flow to track critical point, use mlp, lr, etc models to predict consistant speed
    def __init__(self, old_gray, new_gray, p0,model_name, model, file_name, device_mlp):
        self.old_gray =  old_gray
        self.new_gray = new_gray
        self.p0 = p0
        video_sequence = re.findall(r'([0-9]*)_.', file_name)[0]

        self.model_name = model_name
        self.model = model
        self.device_mlp = device_mlp

        if self.model_name == 'mlp':
            state_path_mlp = '../pt_files/mlp_video' + video_sequence + '.pt'
            self.model = load_model(state=state_path_mlp, device=device_mlp)
            torch.no_grad()
            self.model.eval().to(device_mlp)
        elif model_name == 'lr':
            self.model_lr = pickle.load(open('../pt_files/model_lr_video' + video_sequence + '.dat', 'rb'))
        elif model_name == 'xgbt':
            self.model_xgbt = pickle.load(open('../pt_files/model_xgbt_video' + video_sequence + '.dat', 'rb'))
        elif model_name == 'catbt':
            self.model_catbt = pickle.load(open('../pt_files/model_catbt_video' + video_sequence + '.dat', 'rb'))
        else: pass ## don't use ML model to predict average speed


    def distance(self, x_old, y_old, x_new, y_new):
        return np.sqrt((x_new - x_old) ** 2 + (y_new - y_old) ** 2)

    def work(self):
        tracks = []  ## for data processing
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, self.new_gray, self.p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]
        y_pred = 0
        pred_time_whole = []  # for counting prediction execution time
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            pred_start = time.time()

            if self.model_name == 'mlp':
                X = torch.Tensor([[old[0], old[1], new[0], new[1]]])
                y_pred = self.model(X.to(
                    self.device_mlp)).cpu().detach().item()  ## convert to scaler in cpu, shape(xxxx,1) to shape(xxxx,)
            else:
                X = np.array([old[0], old[1], new[0], new[1]]).reshape(1, -1)  # convert to 2d (1,4)

                if self.model_name == 'lr':
                    y_pred = self.model.predict(X)[0]
                elif self.model_name == 'xgbt':
                    y_pred = self.model.predict(X)[0]
                elif self.model_name == 'catbt':
                    y_pred = self.model.predict(X)[0]
                else: y_pred = 0 ##don't predict average speed
            pred_time_whole.append(time.time()-pred_start)
            dist_new_old = self.distance(old[0], old[1], new[0], new[1])
            if dist_new_old < 1 or dist_new_old > 20: continue  ## filter abnormal data
            if y_pred > 10 or y_pred < -10: continue
            a, b = new.ravel()
            c, d = old.ravel()
            tracks.append([a,b,c,d, y_pred])
        p0 = good_new.reshape(-1, 1, 2)
        pred_time = np.sum(pred_time_whole)
        return tracks, p0, pred_time

def main(model_name, file_name, device_mlp):
    n=0
    input_video = os.path.join(root, file_name)
    color = np.random.randint(0, 255, (100, 3))
    cap = cv.VideoCapture(input_video)
    (ret, old_frame) = cap.read()
    mask = np.zeros_like(old_frame)

    if model_name == 'mlp':
        state_path_mlp = '../pt_files/mlp_video1' + '.pt'
        model = load_model(state=state_path_mlp, device=device_mlp)
        torch.no_grad()
        model.eval().to(device_mlp)

    spds = []
    speed_list = []
    fps_list = []
    round = 0
    speed_this_frame =0
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    while True:
        ## puase and resuem
        key1 = cv.waitKey(10)
        # if key1 == ord('k'):
        #     continue ## forward
        if key1 == ord('p'):
            while (True):
                key2 = cv.waitKey(2)
                if key2 == ord('o'):
                    break
        start = time.time()
        (ret, frame) = cap.read()

        n += 1
        if n % 120 == 0:  ## reset every 4 secs.
            p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            round += 1
            mask = np.zeros_like(old_frame)  ## every 120 frames refresh the tracking line
        new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        op = optical_flow(old_gray, new_gray, p0,model_name, model, file_name, device_mlp)
        tracks, p0, pred_time = op.work()  ## p0 is contineuously updated

        old_gray = new_gray
        for j, tr in enumerate(tracks):
            a,b,c,d,y_pred = tr
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
            spds.append(abs(y_pred))
            # cv.putText(frame, 'speed: %.2f' % y_pred, (int(a), int(b)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 55, 255), 2)
        if (len(spds) > 0): speed_this_frame = np.mean(spds)
        speed_list.append(speed_this_frame)
        speed = np.mean(speed_list[-30:])  ## latest 1 sec
        # cv.putText(frame, 'speed: %.2f' % speed, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 55, 255), 2)

        img = cv.add(frame, mask)
        cv.imshow('optical flow', mask)  #
        cv.imshow('frame', img)
        #     out.write(img)
        k = cv.waitKey(2) & 0xff
        if k == 27:
            break
        fps = 1/(time.time()-start)
        fps_list.append(fps)
        # Now update the previous frame and previous points
    fps = np.mean(fps_list[10:])
    print(f'Average fps: {fps}.')

if __name__ == '__main__':
    root = os.path.join(os.environ['HOME'], './Documents/datasets/traffic/sh')
    file_name = '1_Relaxing_highway_traffic.mp4'
    file_name = '2_night_traffic_shanghai_guangfuxilu_202308102030_720.mp4'
    file_name = '3_traffic_shanghai_jinshajianglu_202308050815_720.mp4'
    file_name = '4_traffic_shanghai_changninglu_202308050830_720.mp4'

    model_name = 'mlp'
    device_mlp = 'cuda'
    main(model_name, file_name, device_mlp)