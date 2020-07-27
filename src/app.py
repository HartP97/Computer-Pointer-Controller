import argparse
import cv2
import os
import time
import logging as log
from face_detection import Fd_Network
from landmark_detection import Lr_Network
from head_pose_detection import Hp_Network
from gaze_estimation import Ge_Network
from input_feeder import InputFeeder
from mouse_controller import MouseController
from general_model import preprocess_image, detect_face, preprocess_hp_output, preprocess_lr_output, preprocess_ge_output

INPUT_STREAM = "../bin/demo.mp4"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # Create the descriptions for the command line arguments
    fdm_desc = "The location of the face-detection model."
    lrm_desc = "The location of the landmark-regression model."
    hpm_desc = "The location of the head-pose-estimation model."
    gem_desc = "The location of the gaze-estimation mode."
    d_desc = "The device name, if not 'CPU', can be GPU, FPGA or MYRIAD"
    ct_desc = "The confidence threshold to use with the bounding boxes"
    i_desc = "Input-type of the Stream, either 'cam' or give video-file-directory"
    flags_desc = "Select from following flags: ffd, flr, fhp, fge (if multiple, enter with single [Space]"\
                 "ffd -> flagFaceDetection, flr -> flagLandmarkRegression"\
                 "fhp -> flagHeadPose, fge -> flagGazeEstimation"

    # Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Create the arguments
    required.add_argument("-fdm", help=fdm_desc, required=True)
    required.add_argument("-lrm", help=lrm_desc, required=True)
    required.add_argument("-hpm", help=hpm_desc, required=True)
    required.add_argument("-gem", help=gem_desc, required=True)
    optional.add_argument("-i", help=i_desc, default='cam')
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    optional.add_argument("-flags", help=flags_desc, nargs='+', default=[], required=False)
    args = parser.parse_args()

    return args

# Function to visualize the selected output of the models (can be selected via -flags command line argument)
def draw_results(frame, cropped_face, coords_face, l_eye_img, r_eye_img, eye_coords, hp_output, gaze_vector, prev_flags, height):

    result_frame = frame.copy
    font = cv2.FONT_HERSHEY_COMPLEX
    c_green = (0, 255, 0)
    c_red = (0, 0, 255)

    if 'ffd' in prev_flags:
        # Draw face-box on frame
        cv2.rectangle(frame, (coords_face[0][0], coords_face[0][1]), (coords_face[0][2], coords_face[0][3]), c_green, 3)

    if 'flr' in prev_flags:
        # Draw eye-box on the cropped face
        cv2.rectangle(cropped_face, (eye_coords[0][0] - 10, eye_coords[0][1] - 10),
                      (eye_coords[0][2] + 10, eye_coords[0][3] + 10), c_green, 2)
        cv2.rectangle(cropped_face, (eye_coords[1][0] - 10, eye_coords[1][1] - 10),
                      (eye_coords[1][2] + 10, eye_coords[1][3] + 10), c_green, 2)

    if 'fhp' in prev_flags:
        # Draw Text of Head-Pose-Angles on frame
        cv2.putText(frame, "Head-Pose Angles: Yaw= {:.2f} , Pitch= {:.2f} , Roll= {:.2f}".format(
            hp_output[0], hp_output[1], hp_output[2]), (20, height-40), font, 1, c_green, 2)

    if 'fge' in prev_flags:
        cv2.putText(frame, "Gaze Coordinates: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(
            gaze_vector[0], gaze_vector[1], gaze_vector[2]), (20, height-80), font, 1, c_green, 2)

        x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
        le = cv2.line(l_eye_img.copy(), (x - w, y - w), (x + w, y + w), c_red, 2)
        cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 150, 255), 2)
        re = cv2.line(r_eye_img.copy(), (x - w, y - w), (x + w, y + w), c_red, 2)
        cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 150, 255), 2)
        cropped_face[eye_coords[0][1]:eye_coords[0][3], eye_coords[0][0]:eye_coords[0][2]] = le
        cropped_face[eye_coords[1][1]:eye_coords[1][3], eye_coords[1][0]:eye_coords[1][2]] = re

    return result_frame


def infer_on_video(args):
    args.ct = float(args.ct)
    input_file = args.i

    # Check if 'cam' or video file was chosen?
    if input_file.lower() == 'cam':
        i_feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_file):
            log.error("Wasn't able to find video file, please correct directory!")
            exit(1)
        i_feeder = InputFeeder(input_type='video', input_file=input_file)

    # Load image/frame of chosen medium
    i_feeder.load_data()
    
    # Initialize the Inference Engine for each model
    fd_plugin = Fd_Network()
    lr_plugin = Lr_Network()
    hp_plugin = Hp_Network()
    ge_plugin = Ge_Network()

    # Load the network models into the IE and get the net input shape
    start_load_time = time.time()
    fd_plugin.load_model(args.fdm, args.d)
    lr_plugin.load_model(args.lrm, args.d)
    hp_plugin.load_model(args.hpm, args.d)
    ge_plugin.load_model(args.gem, args.d)
    total_load_time = time.time() - start_load_time
    log.info("Time it took to load all models: " + str(total_load_time))

    mouse_controller = MouseController('medium', 'fast')

    # Get net input shape of models
    fd_net_input_shape = fd_plugin.get_input_shape()
    lr_net_input_shape = lr_plugin.get_input_shape()
    hp_net_input_shape = hp_plugin.get_input_shape()
    # Currently not used as it didn't return the needed shape correctly for gaze estimation
    # ge_net_input_shape = ge_plugin.get_input_shape()

    # frame_count for FPS calc and start_inf_time, to calc total inference time
    frame_count = 0
    start_inf_time = time.time()
    # Process frames until the video ends, or process is exited
    for ret, frame in i_feeder.next_batch():
        if not ret:
            break

        frame_count += 1
        key_pressed = cv2.waitKey(60)

        height, width = frame.shape[:2]

        ##### FACE-DETECTION #START#
        # Pre-process the frame
        fd_frame = preprocess_image(frame, fd_net_input_shape[3], fd_net_input_shape[2], "face-detection")

        # Perform inference on the frame
        fd_plugin.async_inference(fd_frame)
        
        # Get the output of inference
        if fd_plugin.wait() == 0:
            result = fd_plugin.extract_output()
            # Get frame with bounding box for face, a cropped version and it's coords
            cropped_face, coords_face = detect_face(frame, result, args, width, height)
            ##### FACE-DETECTION #END#

            ##### LANDMARK REGRESSION MODEL #START#
            lr_frame = preprocess_image(cropped_face, lr_net_input_shape[3], lr_net_input_shape[2], "landmark-regression")
            lr_plugin.async_inference(lr_frame)

            if lr_plugin.wait() == 0:
                lr_result = lr_plugin.extract_output()
                l_eye_img, r_eye_img, eye_coords = preprocess_lr_output(lr_result, cropped_face)
            ###### LANDMARK REGRESSION MODEL #END#

            ##### HEAD POSE MODEL #START#
            hp_frame = preprocess_image(cropped_face, hp_net_input_shape[3], hp_net_input_shape[2], "head-pose")
            hp_plugin.async_inference(hp_frame)

            if hp_plugin.wait() == 0:
                hp_result = hp_plugin.extract_output()
                hp_output = preprocess_hp_output(hp_result)
            ##### HEAD POSE MODEL #END#

            ##### GAZE AND MOUSE #START#
            # Hard-coded value because net-input-shape didn't return correctly for the gaze-estimation model
            p_l_eye_img = preprocess_image(l_eye_img, 60, 60, "gaze-estimation")
            p_r_eye_img = preprocess_image(r_eye_img, 60, 60, "gaze-estimation")
            # Perform inference on eye images and head pose output
            ge_plugin.async_inference(p_l_eye_img, p_r_eye_img, hp_output)

            if ge_plugin.wait() == 0:
                ge_result = ge_plugin.extract_output()
                mouse_coord, gaze_vector = preprocess_ge_output(ge_result, hp_output)
            ##### GAZE AND MOUSE #END#

            # Draw on frame if at least one flag was entered via command line
            if len(args.flags) != 0:
                draw_results(frame, cropped_face, coords_face, l_eye_img, r_eye_img, eye_coords, hp_output, gaze_vector, args.flags, height)

            cv2.imshow("cropped", cropped_face)
            # cv2.imshow("Left Eye", l_eye_img)
            # cv2.imshow("Right Eye", r_eye_img)
            cv2.imshow("frame", frame)

            if frame_count % 5 == 0:
                mouse_controller.move(mouse_coord[0], mouse_coord[1])
            
        # Break if escape key pressed
        if key_pressed == 27:
            break

    total_inf_time = time.time() - start_inf_time
    fps = (frame_count / total_inf_time)
    log.info("Total-Inference-Time:" + str(total_inf_time))
    log.info("FPS: " + str(fps))
    # Release the capture and destroy any OpenCV windows
    i_feeder.close()
    cv2.destroyAllWindows()

def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()


'''
### First source the OpenVino Env:
source /opt/intel/openvino/bin/setupvars.sh

### Then run, one of the following examples:
For VIDEO-Input:
python3 app.py -fdm models/face-detection-adas-binary-0001.xml -lrm models/landmarks-regression-retail-0009.xml -hpm models/head-pose-estimation-adas-0001.xml -gem models/gaze-estimation-adas-0002.xml -flags ffd flr fhp fge  -i ../bin/demo.mp4
For WEBCAM-Input:
python3 app.py -fdm models/face-detection-adas-binary-0001.xml -lrm models/landmarks-regression-retail-0009.xml -hpm models/head-pose-estimation-adas-0001.xml -gem models/gaze-estimation-adas-0002.xml -i cam
'''

