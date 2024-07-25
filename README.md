## Citation: 

Yu Liu, Kyoung Don Kang, "AROD: Adaptive Real-Time Object Detection
Based on Pixel Motion Speed," In Proceedings of IEEE Vehicular Technology Conference (VTC2024-Fall), Washington DC, October 7-10, 2024.

## Setup data folder:
- creat the folder: ~'/Documents/datasets/traffic/sh' to put videos below

## Download the video
- file_name = '1_Relaxing_highway_traffic.mp4'  
  - https://www.youtube.com/watch?app=desktop&v=nt3D26lrkho
- file_name = '2_night_traffic_shanghai_guangfuxilu_202308102030_720.mp4'  
  - https://www.youtube.com/watch?v=ZI-tcEbklks&t=162s
- file_name = '3_traffic_shanghai_jinshajianglu_202308050815_720.mp4'  
  - https://www.youtube.com/watch?v=AeszBK_mKFg&t=135s
- file_name = '4_traffic_shanghai_changninglu_202308050830_720.mp4' 
  - https://www.youtube.com/watch?v=Mx1OwCLqVfY&t=68s


## Execute
- /models/optical_flow.py : visualize the optical flow algorithm
- src/main.py:  execute AROD, revise below variables in this code in case need
  - file_name:  to select videos 
  - model_name: models for measuring traffic flow speed
  - od_model_name: models for object detection
