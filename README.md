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

## The Pipeline

Flow of the data inside between the model and inside the application was the following:


Short Introduction / Description
Project Setup and Installation
How to run a demo
The command line options
Explanation of the directory structure and overview of the files used in the project

Benchmarking
- results for models of different precisions
- Discussion of the difference in the results among the models with different precisions (for instance, are some models more accurate than others?)
