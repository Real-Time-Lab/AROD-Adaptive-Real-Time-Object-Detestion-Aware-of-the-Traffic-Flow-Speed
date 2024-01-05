import profile
import warnings
import pandas as pd
import cv2 as cv
warnings.filterwarnings("ignore")
import sys
sys.path.append('../models')  # sys.path.insert(0, '../models')
from optical_flow import optical_flow
from mlp import load_model
import math
import os
import torch
import numpy as np
import multiprocessing as mp
import mp_exception as mp_new

import time
# import erd_cnn
import yolov5
import ssd
import profiler
import re
import pickle

# import efficientDet  ## comment it out when apply yolov5 since it conflicts with "import yolov5"

root = os.path.join(os.environ['HOME'], './Documents/datasets/traffic/sh')

## file name associated with the trained model, need start with 1,2,3,4 to connect
file_name = '1_Relaxing_highway_traffic.mp4'
file_name = '2_night_traffic_shanghai_guangfuxilu_202308102030_720.mp4'
file_name = '3_traffic_shanghai_jinshajianglu_202308050815_720.mp4'
# file_name = '4_traffic_shanghai_changninglu_202308050830_720.mp4'

model_name = 'mlp'  ## mlp, lr, xgbt, catbt, None, none means no ML model appled
device_mlp = 'cpu'
profiling_num = 20  ## in seconds
od_model_name = 'yolov5'  ## 'yolov5', 'ssd', 'efficientDet'
# od_model_name = 'ssd'  ## 'yolov5', 'ssd', 'efficientDet'
# od_model_name = 'efficientDet'  ## 'yolov5', 'ssd', 'efficientDet'
device_od = 'cuda'


video_sequence = re.findall(r'([0-9]*)_.',file_name)[0]  # to extract the first letter(number) of video name so to allocation the associated predicting model
video = os.path.join(root, file_name)  # home
model_od_zoo = ['yolov5', 'ssd', 'efficientDet']
interval = 2
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=20,  ## set 10 in paper
                      qualityLevel=0.1,
                      minDistance=2,
                      blockSize=7)
# Parameters for lucas kanade optical flow
# maxLevel
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def array_to_mask(arry, threshold):
    arry[arry < threshold] = 0
    arry[arry >= threshold] = 1
    return arry

# def adapt_fps(speed, interval): ## pid
#     pass
def opticalFlow_od(config, pipe, queue):
    device_od = config['device']  ##'cuda'
    model_name_od = config['arch']  ##yolov5'  # 'ssd' or 'yolov5' or efficientDet
    con_a, con_b= pipe  ## con_a: send, con_b: recieve
    queue = queue
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  ###Roy

    ##### set optical flow parameters ##############
    data = pd.DataFrame()
    frame_id = 0
    n = 0
    color = np.random.randint(0, 255, (100, 3)) ## color for displaying tracking lines
    ## set video parameters
    video_capturer = cv.VideoCapture(video)
    fps_video = video_capturer.get(cv.CAP_PROP_FPS)
    frame_width = int(video_capturer.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capturer.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    (ret, old_frame) = video_capturer.read()
    print(f'Video original fps: {fps_video}, size: {frame_height} x {frame_width}.')

    mask = np.zeros_like(old_frame)
    spds = []
    round = 0
    speed_this_frame = 0
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)  ## initialize old_gray
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params) ## initialize p0
    sizes = (frame_width, frame_height)
    ##^^^^^^^ above is to set optical flow parameters ^^^^^^^^^####

    latency_od, latency_whole  = [], []
    speed_list, time_list,interval_list, fps_list = [], [], [], []
    speeds = []
    time_of_list , time_pred_list = [],[] ## for counting optical flow execution time
    # ============== mask =============
    # to_mask = True
    ##########  a. mask by image
    # mask_blank = cv.imread('../image/mask_360x640.png')  # 360x640.png, _500x400
    # mask_blank =  cv.imread('./image/mask_180x320_rightside.png')  # 360x640.png, 180x320_rightside, _500x400
    # mask_resized = cv.resize(mask_blank, sizes, interpolation=cv.INTER_CUBIC)
    ########## b. mask by hgight ##########
    # top= 0.4
    # bottom= 0.9

    # ==============
    ################### OPtical flow above #######################
    # speed_interval={(10,1000):1,(5,10):2, (4,5):3, (3,4):4, \
    #                 (2,3):5,(0,2):6}

    speed_interval={(10,1000):1,(5,10):2, (2.5,3.3):3, (2,2.5):4, \
                    (1.7,2):5,(0,1.7):6}

    assert model_name_od in model_od_zoo, f'Model name is not correct, shall be one of {model_od_zoo}'
    if model_name_od == 'efficientDet': import efficientDet  ## do this way cause it conflicts with "import yolov5"
    model_od = eval(model_name_od + '.load_model(device_od)')  ## load model in yolov5.py or ssd.py or efficientNet.py

    #1.############# load velocity prediction models ####################
    if model_name == 'mlp':
        state_path_mlp = '../pt_files/mlp_video' + video_sequence + '.pt'
        model = load_model(state=state_path_mlp, device=device_mlp)
        torch.no_grad()
        model.eval().to(device_mlp)
    elif model_name == 'lr':
        model = pickle.load(open('../pt_files/model_lr_video' + video_sequence + '.dat', 'rb'))
    elif model_name == 'xgbt':
        model = pickle.load(open('../pt_files/model_xgbt_video' + video_sequence + '.dat', 'rb'))
    elif model_name == 'catbt':
        model = pickle.load(open('../pt_files/model_catbt_video' + video_sequence + '.dat', 'rb'))
    else:
        pass  ## don't use ML model to predict average speed
   #################################################################

    new_start = time.time()
    while (video_capturer.isOpened()):

        # piece of code to support pause(p) and resume(o) video, comment is when implement
        key1 = cv.waitKey(10)
        if key1 == ord('p'):
            # print(f'At {n}th frame, video is paused.')
            while (True):
                key2 = cv.waitKey(2)
                if key2 == ord('o'):
                    break

        ## count fps
        if n==10:
            new_start = time.time()  ## count time stamp on the 10th frame

        ## if profiler notice end, then quit
        if con_b.poll():
            msg = con_b.recv()
            con_b.close()
            if msg == 'done':
                print('Optial_od: Get notice from profiller to end!')
                break
        try:
            start = time.time()
            print(f"Frame progress: {n}.", end="\r")
            is_opened, frame = video_capturer.read()  # (720, 1280, 3)， H,W,C (BGR)

            if is_opened:
                frame_id += 1
                od_on = True
                t2 = 0
                n += 1
                start_optical_flow = time.time()  ## for counting optical flow execution time

                #2.###### optical flow, Comment below to bypass optical flow, Note: MLP or other ML model is porcessed inside
                if n % 120 == 0:  ## reset every 4 secs. comment below codes to resume optical flow+mlp
                    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                    round += 1
                    mask = np.zeros_like(old_frame)  ## every 120 frames refresh the tracking line
                new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                op = optical_flow(old_gray, new_gray, p0, model_name, model, file_name, device_mlp) ## Instantize optical flow func., from optical_flow import optical_flow
                tracks, p0 , pred_time = op.work()  ## p0 is contineuously updated
                old_frame = frame.copy()
                old_gray = new_gray
                for j, track in enumerate(tracks):
                    a, b, c, d, y_pred = track
                    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
                    spds.append(abs(y_pred))
                    cv.putText(frame, 'speed: %.2f' % y_pred, (int(a), int(b)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                               (0, 55, 255),
                               2)
                speed_this_frame = np.mean(spds) if (len(spds) > 0) else 0

                # speed_this_frame = 0 ### Roy, for testing
                speeds.append(speed_this_frame)
                speed = np.mean(speeds[-30:])  ## latest 1 sec
                cv.putText(frame, 'speed: %.2f' % speed, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 55, 255), 2)

                ## calculate frame od interval based on the average speed
                for (low, upper), intv in speed_interval.items():
                    if speed >= low and speed < upper:
                        interval = intv
                time_of_list.append(time.time()-start_optical_flow)  # to count time of optical flow
                time_pred_list.append(pred_time)
                ########################################################################################
                interval = 1 ###
                # interval = 1000000 ## Roy: for testing nammually set interval, comment/uncomment, can set a huge value to bypass OD
                if key1 == ord('s'):
                    interval = 5
                    while (True):
                        key2 = cv.waitKey(2)
                        if key2 == ord('o'):
                            interval =1
                            break

                if n % interval == 0:  # interval =2: detect every 2 frames
                    ## timer for start of stage2
                    if device_od == 'cuda':
                        starter.record()
                    else:
                        start_s2 = time.time()
                    if od_on:  # object on the road detected
                        if model_name_od == 'yolov5':
                            res_od = yolov5.predict(model_od, frame)
                        elif model_name_od == 'ssd':
                            bboxes, classes, confidences = ssd.predict(model_od, frame, device_od)
                        else:  ## efficientDet
                            ori_imgs, framed_imgs, framed_metas = efficientDet.preprocess_video(frame, max_size=512)
                            res_od = efficientDet.predict(model_od, framed_imgs, framed_metas, device_od)
                        #
                        # start_imgshow = time.time()
                        if model_name_od == 'yolov5':
                            frame = yolov5.display(res_od)  ## or: frame = np.squeeze(res_od.render())
                        elif model_name_od == 'ssd':
                            if len(classes) > 0:
                                for i in range(len(classes)):
                                    xl, yl, xr, yr = int((bboxes[i][0]) * frame_width), int((bboxes[i][1]) * frame_height), \
                                        int((bboxes[i][2]) * frame_width), int((bboxes[i][3]) * frame_height)
                                    cv.rectangle(frame, (xl, yl), (xr, yr), (255, 0, 0), 1)
                                    # cv.putText(frame, str(classes[i]), (xl, yl), 1, 1, (0, 255, 0))
                            else:
                                pass
                        else:  ## efficientDet
                            frame = efficientDet.display(res_od, ori_imgs)

                    else:
                        t2 = 0
                        cv.putText(frame, '%s' % 'Empty Road', (780, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (155, 255, 55),2)

                    ## timer for end of stage2
                    if device_od == 'cuda':
                        ender.record()
                        torch.cuda.synchronize()  ###
                        t2 = starter.elapsed_time(ender) / 1000
                    else:
                        t2 = time.time() - start_s2

                ## record instant data
                time_list.append(time.time())
                latency_od.append(t2)
                latency = time.time() - start
                latency_whole.append(latency)
                fps= 1/latency
                fps_list.append(fps)
                # speed_list.append(speed)
                speed_list.append(0)
                interval_list.append(interval)
                sampling_num = 5
                queue.put({'traffic_spped': np.mean(speed_list[-sampling_num:]),\
                        'interval':np.mean(interval_list[-sampling_num:]),\
                        'fps': np.mean(fps_list[-sampling_num:]), \
                        'latency': np.mean(latency_whole[-sampling_num:])})  # latest 5 to average
                frame = cv.add(frame, mask)  ## add traking line to show
                cv.imshow('Result', frame)

                ## below paragraph is for grabbing images for composign paper, one time use only
                # nn = 0
                # if n>7350 and n< 7400:  ##7373
                #     if nn % 2 ==0: ## save every 5 frames
                #         cv.imwrite('/home/royliu/Documents/datasets/traffic/temp/frame_'+str(n)+'.png', frame)
                #     nn+=1
                # if n>7500: break

                key = cv.waitKey(2) & 0xFF  # or, waitKey(int(1000/fps_video))
                if key == 27:  # 27: ESC to quite ， q: ord('q')
                    is_open = False  # end of video
                    break
            else:
                break

        except Exception as e:  # press stop to break
            print(e)
            print('Stop surveillance!')
            break

    # video_capturer.release()  ##if hide imshow function, this shall be hidden as well
    # cv.destroyAllWindows()  ##if hide imshow function, this shall be hidden as well

    con_a.send('stop')
    con_a.close()
    # save_log(data, file_name)  ## path(dir) is defined in my_utils.py    # print(cpu_usg_dict)

    # data['time_stamp'] = time_list
    data['traffic_speed'] = speed_list
    data['interval'] = interval_list
    data['fps'] =fps_list
    data['latency'] = latency_whole
    # data['latency_od'] = latency_od

    ## save dataframe
    # data.to_csv(os.path.join(data_dir,'./result/surveillance', (file_name+'.csv')), index=False)
    print('FPS:', 1/((time.time()-new_start)/(n-10)))  ## align with codes in line 131
    mlp_device_name = ',in ' + device_mlp + '.' if model_name == 'mlp' else ''
    print(f'Optical flow (including prediction) execution time: {np.mean(time_of_list[10:])} sec.')
    print(f'Prediction with {model_name}, exexution time:{np.mean(time_pred_list[10:])} sec. {mlp_device_name}')

def main():
    torch.multiprocessing.set_start_method('spawn')
    # profiling_num = 20 ## in seconds
    # od_model_name = 'yolov5'  ## 'yolov5', 'ssd', 'efficientDet'
    # device_od = 'cuda'
    queue= mp.Queue()
    pipe = mp.Pipe()
    config= {'profiling_num':profiling_num, 'arch':od_model_name, 'device': device_od}
    p1= mp.Process(target= opticalFlow_od, args=(config, pipe,),kwargs=dict(queue=queue))
    p2= mp.Process(target= profiler.profile, args=(config,pipe),kwargs=dict(queue=queue))

    # process_list =[p1,p2] ## p2: do profile
    process_list = [p1]
    try:
        for p in process_list:
            p.start()
        for p in process_list:
            p.join()
        for p in process_list:
            p.terminate()
    except:
        for p in process_list:
            p.terminate()

if __name__=='__main__':
    main()


