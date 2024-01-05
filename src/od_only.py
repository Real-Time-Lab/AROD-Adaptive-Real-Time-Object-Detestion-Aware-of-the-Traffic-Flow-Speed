######################################################
# object detection only
#  This file is modified based on erd_od.py by taking off the s1
######################################################
# !/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import sys
sys.path.append('../models') # sys.path.insert(0, '../models')
import cv2
import os
import torch
import numpy as np
import torchvision.transforms as transforms  # process image
import time
# import erd_cnn
import yolov5
import ssd
# import efficientDet  ## comment it out when apply yolov5 since it conflicts with "import yolov5"

###########  setting #########################################
# device_s1 = 'cuda'  # when use cpu to execute
device_s2 = 'cuda'
model_name_s2 = 'yolov5'  # 'ssd' or 'yolov5' or efficientDet
####################################################


interval = 1
# root = 'C:/dataset/traffic' ## win10
root = os.path.join(os.environ['HOME'],'./Documents/datasets/traffic/sh')
file_name = '1_Relaxing_highway_traffic.mp4'  ## both lab and local
# file_name = '2_traffic_shanghai_guangfuxilu_202308031400_720.mp4'
# file_name = '3_traffic_shanghai_jinshajianglu_202308050815_720.mp4'
# file_name = '4_traffic_shanghai_changninglu_202308050830_720.mp4'
# file_name = '4_night_traffic_shanghai_changninglu_202308101930_720.mp4'   ## video 5
# file_name ='3_night_traffic_shanghai_jinshajianglu_202308102000_720.mp4' ## video 6


model_s2_zoo = ['yolov5','ssd', 'efficientDet']
assert model_name_s2 in model_s2_zoo, f'Model name is not correct, shall be one of {model_s2_zoo}'
# model_s1 = erd_cnn.load_model(device_s1)
if model_name_s2 == 'efficientDet': import efficientDet  ## do this way cause it conflicts with "import yolov5"
model_s2 = eval(model_name_s2+'.load_model(device_s2)')
video = os.path.join(root,file_name )  # home

video_capturer = cv2.VideoCapture(video)
fps_video = video_capturer.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
erd_image_size= (640,360)
frame_id = 0
torch.no_grad()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  ###Roy
exe_time_gpu, exe_time_cpu, duration, duration_whole = [], [], [], []
i = 0
# video_h, video_w, c = 720, 1280, 3
results_erd, latency_s1, latency_s2, latency_whole = [], [], [], []
# data = pd.DataFrame(columns = ['empty_road_detection_result', 'latency_stage1', 'latency_stage2', 'latency_whole'])
data = pd.DataFrame(columns = [ 'latency_od', 'latency_whole'])
n= 0
# speed =100
start_whole = time.time()
while (video_capturer.isOpened()):
    print(f"Frame progress: {n}.", end="\r")
    n+=1
    # if n>100: break
    is_opened, frame = video_capturer.read()  # (720, 1280, 3)， H,W,C (BGR)
    start = time.time()
    # if device_s1 == 'cuda':
    #     starter.record()

    if is_opened == True:
        frame_id += 1
        t1 = 0
        res_s1 = 1

        if i % interval == 0:  # interval =2: detect every 2 frames
            ## timer for start of stage2
            if device_s2 == 'cuda':
                starter.record()
            else: start_s2 = time.time()
            if res_s1 == 1:  # object on the road detected
                if model_name_s2 == 'yolov5':
                    res_s2 = yolov5.predict(model_s2, frame)
                elif model_name_s2 == 'ssd':
                    bboxes, classes, confidences = ssd.predict(model_s2, frame, device_s2)
                else: ## efficientDet
                    ori_imgs, framed_imgs, framed_metas = efficientDet.preprocess_video(frame, max_size=512)
                    res_s2 = efficientDet.predict(model_s2, framed_imgs, framed_metas, device_s2)

                start_imgshow = time.time()
                # cv2.putText(frame, '%s' % 'Nonempty Road', (780, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if model_name_s2 == 'yolov5':
                    # frame = np.squeeze(res_s2.render())
                    frame = yolov5.display(res_s2)  ## or: frame = np.squeeze(res_s2.render())
                elif model_name_s2 == 'ssd':
                    if len(classes) > 0:
                        for i in range(len(classes)):
                            xl, yl, xr, yr = int((bboxes[i][0]) * frame_width), int((bboxes[i][1]) * frame_height), \
                                             int((bboxes[i][2]) * frame_width), int((bboxes[i][3]) * frame_height)
                            cv2.rectangle(frame, (xl, yl), (xr, yr), (255, 0, 0), 1)
                            cv2.putText(frame, str(classes[i]), (xl, yl), 1, 1, (0, 255, 0))
                    else:
                        pass
                else: ## efficientDet
                    frame = efficientDet.display(res_s2, ori_imgs)

            else:
                t2 = 0
                # cv2.putText(frame, '%s' % 'Empty Road', (780, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (155, 255, 55),2)
            cv2.imshow('Result', frame)

            ## timer for end of stage2
            if device_s2 == 'cuda':
                ender.record()
                torch.cuda.synchronize()  ###
                t2 = starter.elapsed_time(ender) / 1000
            else:
                t2 = time.time() - start_s2

            results_erd.append(res_s1)
            latency_s1.append(t1)
            latency_s2.append(t2)
            latency_whole.append(time.time() - start)

            key = cv2.waitKey(2) & 0xFF  # or, waitKey(int(1000/fps_video))
            #         video_writer.write(frame)  # save to video
            if key == 27:  # 27: ESC to quite ， q: ord('q')
                is_open = False  # end of video
                break
        i+=1
    else:
        break
# video_writer.release()
video_capturer.release()
cv2.destroyAllWindows()

print('FPS:', n/(time.time()-start_whole))
# data['empty_road_detection_result'] = results_erd
# data['latency_stage1'] =  latency_s1
data['latency_od'] =  latency_s2
data['latency_whole'] =  latency_whole

file_name = 'od_only_'+ model_name_s2+'_'+device_s2
# data.to_csv(os.path.join('../result/raw_data', (file_name+'.csv')), index=False)
# print(data)
