# Computer-Pointer-Controller

In this project the gaze detection model is used to control the mouse pointer of your computer. The mouse pointer changes position accordingly to the estimated gaze of the users's eyes. It demostrates the ability to run multiple models in the same machine and coordinate the flow of data between those models.

### How it works

The application uses the InferenceEngine API from Intel's OpenVino ToolKit. The gaze estimation model requires three inputs:

- The head pose 
- The left eye image
- The right eye image

To get these inputs for the gaze estimation model, the following pre-trained OpenVino models have been used:

- [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Gaze Estimation](https://docs.openvinotoolkit.org/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

## The Pipeline

Flow of the data inside between the model and inside the application was the following:

![data_pipeline](https://github.com/HartP97/Computer-Pointer-Controller/blob/master/result_images/data_pipeline.png)


## Project Setup and Installation

1. **Step: Install [OpenVino Toolkit v2020.1](https://docs.openvinotoolkit.org/latest/)** but be sure to download all pre-requisites first (to safe trouble).
2. **Step: Clone this repository**
3. **Step: Setup virtual environment**, can be achieved by using the command: `virtualenv venv`, if you are not familiar with creating a virtual environment, I recommend following guide: [Click](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
4. **Step: Download** the following 4 **pre-trained models**
- [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Gaze Estimation](https://docs.openvinotoolkit.org/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
- Download example command: `python3 <openvino dir>/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"`
5. **Step: Install** all the necessary **libraries/dependencies** with the command `pip install requirements.txt` (on macOS: use `pip3` instead of `pip`)

## Directory Structure

![project_structure](https://github.com/HartP97/Computer-Pointer-Controller/blob/master/result_images/project_structure.png)

**main-directory**:
- **bin**: currently only contains the example video, which was part of the atarting project of Udacity
- **README.md**: this document you are currently reading
- **requirements.txt**: required librararies/dependecies that have to be installed
- **src**: 
  - **app.py**: Main file of the application that loads, runs, connects all the models and calculates/displays the results
  - **face_detection.py**
  - **gaze_estimation.py**
  - **general_model.py**: Contains several pre-/post-processing functions for different models
  - **head_pose_detection.py**
  - **input_feeder.py**: Used to load video or webcam stream
  - **landmark_detection.py**
  - **models**: Directory contains the above mentioned models (not part of this GitHub Repo, need to be downloaded as described above)
  - **mouse_controller.py**: used to move the mouse based on the final results of the gaze estimation model
  
## Command line options
The file `app.py` has following command line options available:
- **-fdm**: The location of the face-detection model (required).
- **-lrm**: The location of the landmark-regression model (required).
- **-hpm**: The location of the head-pose-estimation model (required).
- **-gem**: The location of the gaze-estimation mode (required).
- **-i**: Input-type of the Stream, either 'cam' or give video-file-directory.
- **-d**: The device name, if not 'CPU', can be GPU, FPGA or MYRIAD.
- **-ct**: The confidence threshold to use with the models.
- **-flags**: Select from following flags: ffd, flr, fhp, fge (if multiple, enter with single [Space]). (ffd -> flagFaceDetection, flr -> flagLandmarkRegression, fhp -> flagHeadPose, fge -> flagGazeEstimation)

## Run the application
If everything was installed correctly, the application can be run with the following command:
`python3 app.py -fdm models/face-detection-adas-binary-0001.xml -lrm models/landmarks-regression-retail-0009.xml -hpm models/head-pose-estimation-adas-0001.xml -gem models/gaze-estimation-adas-0002.xml -flags ffd flr fhp fge  -i ../bin/demo.mp4`
(Also see above mentioned command line options to achieve different results)

## Benchmarking
When comparing FP16, FP32 and FP32-INT, I have focused on Model-Load-Time, Inference Time and Frames per Second (note: the face-detection model only offered FP32-INT1).

### Model-Load-Time
![model_load_time](https://github.com/HartP97/Computer-Pointer-Controller/blob/master/result_images/model_load_time.png)
### Inference Time
![inference_time](https://github.com/HartP97/Computer-Pointer-Controller/blob/master/result_images/inference_time.png)
### Frames per Second
![fps](https://github.com/HartP97/Computer-Pointer-Controller/blob/master/result_images/frame_per_second.png)

### Comparison
#### FP16
- model-load-time: 0.44661 s
- inference time: 23.5582 s
- fps: 2.504
#### FP32
- model-load-time: 0.5140 s 
- inference time: 23.2349 s
- fps: 2.539
#### FP32-INT 8
- model-load-time: 2.1982 s
- inference time: 23.5177 s 
- fps: 2.509

### Results
- First we can notice that the model-load-time is the lowest for FP16, which makes sense because it is the lowest precision in our comparison (lower precision -> lower accuracy)
- therefore it is not surprising that FPR32-INT 8 takes the longest to load up, as higher precisions lead to higher weight of the model (+ larger model file)
- FPS and Inference Time are pretty similar for the three models (within error deviation)
- Discussion of the difference in the results among the models with different precisions (for instance, are some models more accurate than others?)

