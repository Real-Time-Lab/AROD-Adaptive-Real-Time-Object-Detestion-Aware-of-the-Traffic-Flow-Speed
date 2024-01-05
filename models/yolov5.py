import torch
import cv2
import numpy as np
import os

def load_model(device):
    if device == 'cuda' and torch.cuda.is_available():
        device_yolov5s = 0  # device = 0： 'cuda'。 or 'cpu'
    else:
        device_yolov5s = 'cpu'
    '''run below commented code for the first time to load yolov5 model'''
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,
                              device=device_yolov5s, force_reload=True)  # yolov5x  or yolov5s
    local_source = os.path.join(os.environ['HOME'], './.cache/torch/hub/ultralytics_yolov5_master')
    assert local_source, 'Model does not exist please run download_models.py to download yolov5 models first.'
    model = torch.hub.load(local_source, 'yolov5s', source='local', pretrained=True,
                           device=device_yolov5s)  ## model is loaded locally

    model.conf = 0.50  # confidence threshold (0-1) 0.52
    model.iou = 0.45  # NMS IoU threshold (0-1). 0.45
    return model

def predict(model, frame):
    return model([frame])
def display(results):
    return np.squeeze(results.render())

def main():
    root = os.path.join(os.environ['HOME'],'./Documents/datasets/traffic/sh')  #ubuntu
    device ='cuda'
    model = load_model(device)
    video = os.path.join(root, '1_Relaxing_highway_traffic.mp44')
    video_capturer = cv2.VideoCapture(video)

    while (video_capturer.isOpened()):
        is_opened, frame = video_capturer.read()  # (720, 1280, 3)， H,W,C (BGR)
        if is_opened:
            results = predict(model, frame)
            frame = display(results)
            cv2.imshow('Result', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            # print('Yolov5 test is ok!')
        if not is_opened: break

    video_capturer.release()
    cv2.destroyAllWindows()

def measure_model_size():
    device = 'cuda'
    from torchsummary import summary
    from thop import profile
    model = load_model(device)
    flops, params = profile(model.to(device), inputs=(torch.rand(1,3, 640, 640).to(device),))
    print(f'Flops: {flops}, Parameters:{params}')
    gflops = flops / 1024/1024/1024
    print(f"GFLOPs: {gflops:.2f}")
    # print(stat(model,(1,1,4)))

if __name__ == '__main__':
    # main()
    measure_model_size()