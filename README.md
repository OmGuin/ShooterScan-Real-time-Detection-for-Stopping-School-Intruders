# ShooterScan: Real-time Detection for Stopping School Intruders
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

![Logo](logo.png)


School shootings have increased at an alarming rate over the last several years from an average of about 30 shootings per year in 2015 to an average of about 300 per year in the last 5 years. While processes and policies are being developed to identify and solve the root cause of the problem, it is imperative to implement immediate measures, such as the improvement of security measures, that can effectively deter such incidences in the shorter term. 
This study develops a method for automated identification of individuals that could potentially be a threat to schools by identifying cues including obvious ones such as people carrying firearms,  and more subjective ones such as a person that appears to be extremely distraught while trying to gain entry to the school building. The study uses an internet of AI models for the identification of these characteristics by utilizing video data obtained from cameras placed in public places such as the school parking lot and near building entrances. Additionally, another AI model is used to perform facial recognition for comparison with a known person database relevant to the facility. The weapon possession, negative emotion presence, and known person database comparison cues are used to inform the decisions to allow the person to enter the building. In the case study performed for validation of this methodology at a school site, since these cues are not in themselves a standalone determinant of the threat, these cues are used to generate alerts that are transmitted to appropriate school personnel for further confirmation and appropriate intervention if necessary. The algorithms were designed and tested to satisfy real time performance objectives. This research thus develops a system that can help make schools more secure. The case study results demonstrate that such security can be achieved without violating individual privacy.


## Documentation
Note: .ipynb files are intended to be run on vast.ai. Python 3.9 is recommended for the notebooks.  
Utilize L40 or Nvidia RTX 5090 GPU for ideal computational performance at low cost.  
Thank you [Vast](https://vast.ai/) for bringing convenience to training.

### Doorbell Cam
Obtain RTSP/Flask stream link
Download YOLO model (or desired model)

We found that installing packages this way solved our issues with dlib and face_recognition:
run ``pip install dlib-bin``
run ``pip install -r requirements.txt``
run ``pip install face_recognition --no-deps``



### CCTV cam
Upload .ipynb, deformable_detr_config.py, and requirements.txt in workspace  
\_base\_ and deformable_detr in root
Train model, save it to .pth, and use it for inference 

Use following ffmpeg command if video size exceeds DETR input size
```
ffmpeg -i test_video.mp4 -s 640x480 -c:a copy test_video_resized.mp4
```

## Authors
- [@Pranavmath](https://github.com/Pranavmath)
- [@Om Guin](https://github.com/OmGuin)



## Acknowledgements
 - [CCTV-Gun](https://github.com/srikarym/CCTV-Gun)
 - [face_recognition](http://github.com/ageitgey/face_recognition)
 - [ultralytics](https://github.com/ultralytics/ultralytics)