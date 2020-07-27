import math
import cv2
import logging as log

'''
File that contains model un/-specific pre-/post-processing functions
'''

def preprocess_image(frame, width, height, model_name):
    try:
        p_frame = cv2.resize(frame, (width, height))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
    except Exception as e:
        log.error("Failed to pre-process image for " + str(model_name) + " model: " + str(e), exc_info=True)
    return p_frame

def detect_face(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    Saving detected coords
    Saving cropped face
    '''
    detections = []
    try:
        for box in result[0][0]: # Output shape is 1x1x100x7
            conf = box[2]
            if conf >= args.ct:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                # save detected coords
                detections.append([xmin, ymin, xmax, ymax])
                # crop just face for later use
                cropped = frame[ymin:ymax, xmin:xmax]
    except Exception as e:
        log.error("Following error occurred when extracting coordinates for face from inference results: " + str(e),
                  exc_info=True)
    return cropped, detections

def preprocess_hp_output(results):

    try:
        final_result = [results['angle_y_fc'][0][0], results['angle_p_fc'][0][0], results['angle_r_fc'][0][0]]
    except Exception as e:
        log.error("Following error occurred during pre-processing output of head-pose: " + str(e), exc_info=True)

    return final_result

def preprocess_lr_output(results, frame):

    try:
        h, w = frame.shape[:2]

        results = results[0]

        # found that +-12 worked pretty well to not crop too much or too less
        l_eye_xmin = int(results[0][0][0] * w) - 12
        l_eye_ymin = int(results[1][0][0] * h) - 12
        r_eye_xmin = int(results[2][0][0] * w) - 12
        r_eye_ymin = int(results[3][0][0] * h) - 12

        l_eye_xmax = int(results[0][0][0] * w) + 12
        l_eye_ymax = int(results[1][0][0] * h) + 12
        r_eye_xmax = int(results[2][0][0] * w) + 12
        r_eye_ymax = int(results[3][0][0] * h) + 12

        l_eye_img = frame[l_eye_ymin:l_eye_ymax, l_eye_xmin:l_eye_xmax]
        r_eye_img = frame[r_eye_ymin:r_eye_ymax, r_eye_xmin:r_eye_xmax]

        eye_coords = [[l_eye_xmin, l_eye_ymin, l_eye_xmax, l_eye_ymax],
                             [r_eye_xmin, r_eye_ymin, r_eye_xmax, r_eye_ymax]]
    except Exception as e:
        log.error("Following error occurred when trying to preprocess landmark-regression outputs: " + str(e),
                  exc_info=True)

    return l_eye_img, r_eye_img, eye_coords

def preprocess_ge_output(ge_outputs, hp_coords):

    try:
        gaze_vector = ge_outputs['gaze_vector'][0]
        angle_r_fc = hp_coords[2]
        sin_r = math.sin(angle_r_fc * math.pi / 180.0)
        cos_r = math.cos(angle_r_fc * math.pi / 180.0)
        x = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r
        y = -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
        mouse_coords = (x, y)
    except Exception as e:
        log.error("Following error occurred during pre-processing output for the gaze-estimation: " + str(e),
                  exc_info=True)

    return mouse_coords, gaze_vector