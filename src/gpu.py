import subprocess
import time
import argparse
parser = argparse.ArgumentParser(description='optical flow with object detection')
parser.add_argument('-c', '--clock', default=615, type=int, metavar='N',
                    help='frequency of graphic core (default: 615)')

args = parser.parse_args()

def get_gpu(arg):
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
    # output[pos]= fmt_to_stmp(output[pos], "%Y/%m/%d %H:%M:%S") # convert date-time to stamp format, format in "%Y/%m/%d %H:%M:%S"
    return output

def set_gpu_pm(mode):  # persistence mode 0: none persistent, 1: persistent
    print('Mode is setting (1 means persistent):', mode)
    p = subprocess.Popen(['sudo', 'nvidia-smi', '-pm', str(mode)], stdout=subprocess.PIPE)  # close_fds= True
    stdout, stderror = p.communicate()
    # print('mode setting:', stdout)
def set_gpu_frq(clock):
    # p = subprocess.Popen(['sudo', 'nvidia-smi', '-lgc', 'clock,clock'], stdout=subprocess.PIPE, shell=True)  # close_fds= True
    p = subprocess.Popen(['sudo', 'nvidia-smi', '-lgc', str(clock)], stdout=subprocess.PIPE)  # close_fds= True

    stdout, stderror = p.communicate()
    # print(stdout)
    output = stdout.decode('UTF-8').strip()
    output = output.split(',')  # split the returned string and convert to list

if __name__ == '__main__':
    gpu_frq = int(get_gpu(['gpu_freq'])[0])
    print('Before setting frequency, GPU frequency is :', gpu_frq)
    set_gpu_pm(1)
    # time.sleep(1)
    set_gpu_frq(args.clock) ## at lab: 210,660, 1920 at home: 1603
    gpu_frq = int(get_gpu(['gpu_freq'])[0])
    print('After setting frequency, GPU frequency is :', gpu_frq)