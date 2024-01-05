'''
    us: is meaning of "user CPU time"
    sy: is meaning of "system CPU time"
    ni: is meaning of" nice CPU time"
    id: is meaning of "idle"
    wa: is meaning of "iowait"
    hi：is meaning of "hardware irq"
    si : is meaning of "software irq"
    st : is meaning of "steal time"
'''

import json
import os
import re
import random
import subprocess
import pandas as pd
import time
from my_utils import save_log
import multiprocessing as mp
from gpu import *

gpu3080_supported_clocks = [210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 465, 480, 495, 510, 525, 540, 555, 570, 585, 600, 615, 630, 645, 660, 675, 690, 705, 720, 735, 750, 765, 780, 795, 810, 825, 840, 855, 870, 885, 900, 915, 930, 945, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080, 1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1215, 1230, 1245, 1260, 1275, 1290, 1305, 1320, 1335, 1350, 1365, 1380, 1395, 1410, 1425, 1440, 1455, 1470, 1485, 1500, 1515, 1530, 1545, 1560, 1575, 1590, 1605, 1620, 1635, 1650, 1665, 1680, 1695, 1710, 1725, 1740, 1755, 1770, 1785, 1800, 1815, 1830, 1845, 1860, 1875, 1890, 1905, 1920, 1935, 1950, 1965, 1980, 1995, \
                             2010, 2025, 2040, 2055, 2070, 2085, 2100]  ## total 127 elements
gpu3080_selected_clocks = [210, 360, 510, 660, 810, 960, 1110, 1260, 1410, 1560, 1710, 1860, 2010]
random.shuffle(gpu3080_selected_clocks)

def cpu_freq():
    dev=json.load(open(r'./device.json','r'))
    cpu_freq_dict ={}
    for i in range(os.cpu_count()):

        cpu_name = 'cpu'+str(i)
        arg = dev['cpu'][cpu_name]['freq_now']
        cpu_freq_dict[cpu_name+'_freq']= subprocess.check_output(['cat', arg]).decode('utf-8').strip()
    return cpu_freq_dict


def cpu_usage():
    p = subprocess.Popen(["top", "n1", "b"], \
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    threads_info = stdout.decode('utf-8').split('\n')
    cpu_usg=threads_info[2].split(',')
    cpu_usg_val= re.findall(r'[0-9]+[.][0-9]+', str(cpu_usg))  # parse value
    cpu_usg_col = re.findall('[a-zA-Z]+', str(cpu_usg))[2:] # [:-2]: take off string of '%Cpu(s)'
#     cpu_usg = [float(value.strip("%"))/float(cpu_usg[0].strip("%"))*100 for value in cpu_usg]
#     thrd_num= len(threads_info) -5
#     cpu_usg_col.append('thrd_num')
#     cpu_usg_val.append(thrd_num)
    cpu_usg_dict = dict(zip(cpu_usg_col, cpu_usg_val))

    return cpu_usg_dict

def date_time():
    s_l = time.localtime(time.time())
    dt = time.strftime("%Y%m%d", s_l)
    tm = time.strftime("%H%M%S", s_l)
    # print(date, tm )
    return dt, tm

def fmt_to_stmp(tm,*arg):
    fmt= "%Y-%m-%d %H:%M:%S" if not arg else arg[0]
    dt, ms =tm.split('.')  # to process with milli-second
    timeArray =time.strptime(dt, fmt)
    timeStamp = time.mktime(timeArray)+int(ms)/1000
    return timeStamp

def grab_gpu_data(arg):
    query_dic = {'time_stamp':'timestamp', 'gpu_name':'name', 'index':'index','gpu_power':'power.draw',\
                 'gpu_freq':'clocks.gr', 'gpu_mem_freq':'clocks.mem','gpu_temp':'temperature.gpu',\
                 'gpu_util%':'utilization.gpu', 'gpu_mem_util%':'utilization.memory',\
                 'gpu_mem_total':'memory.total', 'gpu_mem_used':'memory.used'}

    pos, i = -1, 0 # the position of time_stamp for converting
    query = ''  #create an empty string
    for item in arg:
        if item == 'time_stamp':
            pos = i
        i+=1
        query+=query_dic[item]+','

    query='--query-gpu='+query[:-1]  # add query string's head and cut tail, which tail is ','
    nvidia_smi = "nvidia-smi"
    p = subprocess.Popen([nvidia_smi, query, "--format=csv,noheader,nounits"], stdout=subprocess.PIPE) # close_fds= True
    stdout,  stderror = p.communicate()
    output = stdout.decode('UTF-8').strip()
    output = output.split(',')  #split the returned string and convert to list
    output[pos]= fmt_to_stmp(output[pos], "%Y/%m/%d %H:%M:%S") # convert date-time to stamp format, format in "%Y/%m/%d %H:%M:%S"
    return dict(zip(arg, output))

# def save_log(data,config):
#     dt, tm = date_time()
#     log_dir = '../result/log/' + dt
#     if not os.path.exists(log_dir): os.mkdir(log_dir)
#     log_file = config + '_'+ dt + tm
#     data.to_csv(os.path.join(log_dir,log_file)+ '.csv')

def profile(config, pipe, queue):
    con_prf_a, con_prf_b = pipe
    queue= queue
    file_name = 'profile_' + config['arch']+'_'+config['device']
    profiling_num = config['profiling_num']
    # con_prf_b.close()  ##  only send message to main, 关闭收到端
    gpu_col = ['time_stamp', 'gpu_power', 'gpu_freq', 'gpu_mem_freq', 'gpu_temp', 'gpu_util%', 'gpu_mem_util%', 'gpu_name']
    gpu_dict = grab_gpu_data(gpu_col)
    cpu_usg_dict = cpu_usage()
    cpu_freq_dict = cpu_freq()

    cpu_col =  list(cpu_usage().keys()) + list(cpu_freq().keys())  # create the cpu, gpu's columns of dataframe
    cpu = pd.DataFrame(columns=cpu_col, index=None)
    gpu = pd.DataFrame(columns=gpu_col, index=None)

    opf_col = ['timestamp','traffic_speed', 'interval', 'fps', 'latency']
    opf = pd.DataFrame(columns=opf_col, index = None)
    col_select = ['gpu_power', 'gpu_freq', 'gpu_mem_freq', 'gpu_temp', 'gpu_util%', 'gpu_mem_util%', 'us', 'sy',
           'cpu0_freq', 'cpu5_freq']

    start = time.time()
    i = 0

    # while True:
    print('profiling_num:', profiling_num)
    delay, duration,  interval_target = 0.0, 0.9, 1 # profile every 1 sec
    time_prev = time.time()
    set_gpu_pm(1)  ## value 1: set gpu mode to be persistent

    k1, k2 =0,0 ##roy
    s = 0
    for i in range(profiling_num):
        try:
            ## optical flow controller data collection
            opf_dict = queue.get()
            opf_cols = opf_dict.keys()
            val = [time.time()]+ list(opf_dict.values())
            opf.loc[i, opf_col]= val
            fps=val[3]  ## fps
            # print('fps:',fps)

            ## cpu data collection
            cpu_usg_dict = cpu_usage()
            cpu_freq_dict = cpu_freq()
            val = list(cpu_usg_dict.values()) + list(cpu_freq_dict.values())
            cpu.loc[i, cpu_col] = val

            ## gpu data collection
            val = list(grab_gpu_data(gpu_col).values())
            gpu.loc[i, gpu_col] = val

            ##### Below codes to creat a delay to meet target profiling interval, e.g. 1 sec. #########
            time_now = time.time()
            duration = time_now - time_prev
            diff = delay+ interval_target-duration
            if diff > 0.01:  # only change delay when need , make sure delay won't be negative value
                delay= delay + (interval_target-duration)
            elif interval_target-duration < 0:
                delay = 0
            if delay > 0: time.sleep(delay)
            time_prev = time_now
            # print(time.time())
            ######## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^########

            # i += 1
            print('\r', i, ' instances.', end='')

            gpu_set_frq= gpu3080_selected_clocks[s] ## s: subsript

            if i %20 == 0 :
                set_gpu_frq(gpu3080_selected_clocks[s])
                s+=1
                if s >= len(gpu3080_selected_clocks): s = 0
            ## adapt fps  to save energy
            # if fps>31:
            #     k1+=1
            #     if k1>2:
            #         set_gpu_frq(210)
            #         k1=0
            # elif fps<30:
            #     k2+=1
            #     if k2>2:
            #         set_gpu_frq(2100)
            #         k2=0

        except:  # press stop to break
            print()
            print('Profiler: Collecting done!')
            data = gpu.join(cpu)
            # save_log(data, file_name)  ## path(dir) is defined in my_utils.py
            break

        if con_prf_b.poll():  ## to recieve stop notice from main (opf)
            msg = con_prf_b.recv()
            con_prf_b.close()
            if msg == 'stop':
                print()
                print('Profiler: Get notice from main to stop!')
                data = gpu.join(cpu)
                # save_log(data, file_name)
                break
        # print(gpu['gpu_freq'])   ### check gpu frequency
    print(f'Data shape of gpu:{gpu.shape}, and cpu: {cpu.shape}.')
    data = opf.join(gpu)
    data = data.join(cpu)
    # save_log(data, file_name)
    print('Profiler: Profiling is ending, notice to main!')
    con_prf_a.send('done')
    con_prf_a.close()

    print('Profiler: Time elapsed: ', time.time() - start, 'sec.')
    # print(data)
    # return 0

if __name__ == '__main__':  # for local testing
    profiling_num = 0  # only for local debugging
    pipe = mp.Pipe()
    queue=mp.Queue()
    device='cuda'
    config = {'profiling_num': profiling_num, 'arch': 'yolov5', 'device': device}
    profile(config, pipe, queue)  ## will block cause it is wait for con.receive