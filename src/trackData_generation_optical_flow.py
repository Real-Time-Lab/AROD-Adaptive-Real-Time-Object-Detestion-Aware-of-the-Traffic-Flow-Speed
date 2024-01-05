# 来自opencv\sources\samples\python\lk_track.py
# 参考： https://zhuanlan.zhihu.com/p/42942198
# http://wjhsh.net/my-love-is-python-p-10447917.html
## Lucas-Kanade method


import numpy as np
import cv2 as cv
import os
import pandas as pd
import sys

sys.path.append('../models')
from mlp import load_model
from my_utils import mask_rect
import torch
import time
import pickle
import re

from sklearn.linear_model import LinearRegression

root = os.path.join(os.environ['HOME'], './Documents/datasets/traffic')


file_name = '1_Relaxing_highway_traffic.mp4'  ## both lab and local
# file_name = '2_traffic_shanghai_guangfuxilu_202308031400_720.mp4'
# file_name = '3_traffic_shanghai_jinshajianglu_202308050815_720.mp4'
# file_name = '4_traffic_shanghai_changninglu_202308050830_720.mp4'
# file_name = 'traffic_shanghai_guangfuxilu_202307311730_720.mp4'
# file_name = 'traffic_shanghai_guangfuxilu_202308050700_720.mp4'
# file_name = 'traffic_shanghai_jinshajianglu_202308050800_720.mp4'

model_name = 'xgbt'  ## mlp, lr, xgbt, catbt
device_mlp = 'cuda'
video_sequence = re.findall(r'([0-9]*)_.',file_name)[0]
model_lr = pickle.load(open('../pt_files/model_lr_video'+video_sequence+'.dat', 'rb'))
model_xgbt = pickle.load(open('../pt_files/model_xgbt_video'+video_sequence+'.dat', 'rb'))
model_catbt = pickle.load(open('../pt_files/model_catbt_video'+video_sequence+'.dat', 'rb'))
state_path_mlp = '../pt_files/mlp_video'+video_sequence+'.pt'
model = load_model(state= state_path_mlp, device = device_mlp)


# ################# option1, on demand mask ####################
# mask_blank = cv.imread('../image/mask_360x640.png')  # 360x640.png, _500x400
# # mask_blank =  cv.imread('./image/mask_180x320_rightside.png')  # 360x640.png, 180x320_rightside, _500x400
# mask_resized = cv.resize(mask_blank, sizes, interpolation=cv.INTER_CUBIC)

################## option2:  mask by rectangle ###############
upper_left = (0.2, 0.0)  # in percentage
lower_right = (0.9, 0.5)  ## in percentage
mask_resized = mask_rect((720, 1080), upper_left, lower_right)  ##  720 *1080, 720*1980?
##############################################################
to_mask = False

#################above: make rect mask##############################

input_video = os.path.join(root, file_name)
# input_video = os.path.join(root, 'Road traffic video for object detection and tracking - 2020 Challenge_720.mp4')
# input_video = os.path.join(root, 'Traffic count, monitoring with computer vision. 4K, UHD, HD..mp4')
cap = cv.VideoCapture(input_video)
# out_video_name =os.path.join(root, 'track_out_video.mp4')   #when need save output video use it

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=20,
                      qualityLevel=0.1,
                      minDistance=2,
                      blockSize=2)
# Parameters for lucas kanade optical flow
# maxLevel
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors 产生随机的颜色值
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
(is_opened, old_frame) = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# out_fps = 30.0
fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
sizes = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
print(sizes)
# out = cv.VideoWriter(out_video_name, fourcc, out_fps, sizes)

speed = 0

n = 0
speed_list, fps_list = [], []
y_pred = 5
old_new = pd.DataFrame(columns=['item', 'old_x', 'old_y', 'new_x', 'new_y', 'round'])
item_list, old_x_list, old_y_list, new_x_list, new_y_list = [], [], [], [], []
round_list = []  ## for data processing
round = 0
alpha = 0.5 * 360


def distance(x_old, y_old, x_new, y_new):
    return np.sqrt((x_new - x_old) ** 2 + (y_new - y_old) ** 2)

def remove_outliers(data_list):
#     data = sorted(data_list)
    q3 =  np.percentile(data_list,75)
    q1 = np.percentile(data_list,25)
    irq = q3-q1
    lower = q1 -1.5*irq
    upper = q3 +1.5*irq
    removed_outliers = [x for x in data_list if x > lower and x < upper]
    return removed_outliers

# model = load_model(device_mlp)
# model.eval().to(device_mlp)
y_pred = 0
speed_list1 =[]
while True:
    speed_this_frame = 0
    start = time.time()
    ## puase and resuem
    key1 = cv.waitKey(2)
    if key1 == ord('p'):
        while (True):
            key2 = cv.waitKey(5)
            if key2 == ord('o'):
                break  ## brek the pause loop

    n += 1
    if n %120 == 0:  ## reset every 4 secs.
        speed_list1 =[]
        round += 1
        mask = np.zeros_like(old_frame)  ## every 120 frames refresh the tracking line
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  ## reset the p0 , start a new calculation

    (is_opened, frame) = cap.read()
    if not is_opened: break  ## end of video playing
    if to_mask: frame = frame * mask_resized
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    #     print(good_new)
    ## draw the tracks
    # threshold_pix_dist = 1500  ## filter outliers
    spds = []  ## in these 2 frames

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        item_list.append(i)
        old_x_list.append(old[0])
        old_y_list.append(old[1])
        new_x_list.append(new[0])
        new_y_list.append(new[1])
        round_list.append(round)

        if model_name == 'mlp':
            X = torch.Tensor([[old[0], old[1], new[0], new[1]]])
            with torch.no_grad():
                y_pred = model(X.to(device_mlp)).cpu().detach().item()  ## convert to scaler in cpu, shape(xxxx,1) to shape(xxxx,)

        else:
            X = np.array([old[0], old[1], new[0], new[1]]).reshape(1,-1)  # convert to 2d (1,4)

            if model_name == 'lr':
                y_pred = model_lr.predict(X)[0]
            elif model_name == 'xgbt':
                y_pred = model_xgbt.predict(X)[0]
            elif model_name =='catbt':
                y_pred = model_catbt.predict(X)[0]

        dist_new_old = distance(old[0], old[1], new[0], new[1])
        if dist_new_old < 1 or dist_new_old > 20: continue  ## filter abnormal data
        if y_pred > 10 or y_pred < -10: continue  ## filter abnormal track
        a, b = new.ravel()
        c, d = old.ravel()

        # frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        cv.putText(frame, 'speed: %.2f' % y_pred, (int(a), int(b)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 55, 255), 2)
        spds.append(abs(y_pred))
        # speed_list1.append(abs(y_pred[0]))
        speed_list1.append(abs(y_pred))
    ## take the outliers
    if (len(spds) > 0):
        speed_this_frame = np.mean(spds)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    fps_list.append(1 / (time.time() - start))
    speed_list.append(speed_this_frame)
    # speed = np.mean(speed_list[-30:])  ## latest 1 sec

    if len(speed_list1) > 20:
        # speed_list1 = speed_list1[-50:]
        fitered_spds = remove_outliers(speed_list1[-20:]) ## the newest 30 tracks
        speed = np.mean(speed_list1)
    else: speed = 0

    ## display
    cv.putText(frame, 'speed: %.2f' % speed, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 55, 255), 2)
    img = cv.add(frame, mask)
    cv.imshow('optical flow', mask)  #
    cv.imshow('frame', img)
    #     out.write(img)
    k = cv.waitKey(2) & 0xff
    if k == 27:
        break

    # print(speed)

old_new.item = item_list
old_new['round'] = round_list
old_new.old_x, old_new.old_y, old_new.new_x, old_new.new_y = \
    old_x_list, old_y_list, new_x_list, new_y_list

# print(old_new)
# old_new.to_csv('../result/bk_opticalFlow_tracks_jinshajiangRoad.csv')
cv.destroyAllWindows()
cap.release()
fps = np.mean(fps_list[10:])
print(f'Average fps: {fps}.')