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

1. **Step:** Install [OpenVino Toolkit v2020.1](https://docs.openvinotoolkit.org/latest/) but be sure to download all pre-requisites first (to safe trouble).
2. **Step:** Clone this repository
3. **Step:** Setup virtual environment, can be achieved by using the command: `virtualenv venv`, if you are not familiar with creating a virtual environment, I recommend following guide: [Click](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
4. **Step:** Download the following 4 pre-trained models
- [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Gaze Estimation](https://docs.openvinotoolkit.org/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
5. **Step:** Install all the necessary libraries/dependencies with the command `pip install requirements.txt` (on macOS: use `pip3` instead of `pip`)


- Project Setup and Installation
- How to run a demo
- The command line options
- Explanation of the directory structure and overview of the files used in the project

Benchmarking
- results for models of different precisions
- Discussion of the difference in the results among the models with different precisions (for instance, are some models more accurate than others?)
