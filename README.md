
# ShooterScan: Real-time Detection for Stopping School Intruders
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

School shootings have been increasing over the last several years with one of
the most recent ones happening in Winder, Georgia. While processes are
developed to identify and solve the root cause of the problem, we also need to
implement short term measures such as the improvement of security
measures. This project develops a method for automated identification of
individuals that could potentially be a threat to schools by identifying cues
such as people carrying a firearm or a person that appears to be extremely
angry while trying to gain entry to the school building. This project uses AI
models for the identification of these characteristics by processing videos from
cameras placed in public places such as the school parking lot and near the
door of the building. A third AI model performs facial recognition. If no threat
related to weapons or negative emotion detection exists, and the person is
identified as a known person by comparing with the school database, the door
is automatically unlocked. Since these cues are not in themselves a
standalone determinant of the threat, these cues are used to generate alerts
that are transmitted to appropriate school personnel for further confirmation
and appropriate intervention if necessary. Functional system prototypes are
developed for video streaming and remote processing in real-time and remote
operation of door locks and then tested on-site at a school. This project thus
develops and demonstrates a system that can help make our schools safer
and more secure without violating individual privacy.

## Documentation
Note: .ipynb files are intended to be run on vast.ai. Utilize L40 or Nvidia RTX 4090 GPU for ideal performance at low cost.
Thank you [Vast](vast.ai) for bringing convenience to training.

### Doorbell Cam
Obtain RTSP/Flask stream link  
Download YOLO model  (or desired model)  
pip install face recognition (need dlib, cmake, etc.)

### CCTV cam
Upload .ipynb  
deformable_detr_config.py in workspace  
requirements.txt in workspace  
_base and deformable_detr in root
Train model to get .pth... Refer to trained .ipynb for 

Use following ffmpeg commands if video size exceeds DETR input size
```
ffmpeg -i 20241203_170237.mp4 -s 640x480 -c:a copy 20241203_170237_resized.mp4

ffmpeg -i 20241203_170144.mp4 -s 640x480 -c:a copy 20241203_170144_resized.mp4

ffmpeg -i 20241203_170119.mp4 -s 640x480 -c:a copy 20241203_170119_resized.mp4

ffmpeg -i 20241203_170054.mp4 -s 640x480 -c:a copy 20241203_170054_resized.mp4

ffmpeg -i 20241203_165932.mp4 -s 640x480 -c:a copy 20241203_165932_resized.mp4
```

## Authors
- [@Om Guin](https://github.com/OmGuin)
- [@Pranavmath](https://github.com/Pranavmath)



## Acknowledgements

 - [CCTV-Gun](https://github.com/srikarym/CCTV-Gun)
