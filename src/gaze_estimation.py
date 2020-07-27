'''
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Ge_Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU"):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        try:
            model_xml = model
            model_bin = os.path.splitext(model_xml)[0] + ".bin"

            # Initialize the plugin
            self.plugin = IECore()

            # Read the IR as a IENetwork
            self.network = IENetwork(model=model_xml, weights=model_bin)

            # Load the IENetwork into the plugin
            self.exec_network = self.plugin.load_network(self.network, device)

            # Get the input layer
            self.input_blob = next(iter(self.network.inputs))
            self.output_blob = next(iter(self.network.outputs))
        except Exception as e:
            log.error("Following error occurred while loading the gaze-estimation model:" + str(e), exc_info=True)

        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        try:
            i_shape = self.network.inputs[self.input_blob].shape
        except Exception as e:
            log.error("Following error occurred when getting input shape for gaze-estimation model: " + str(e),
                      exc_info=True)
        return


    def async_inference(self, l_eye_img, r_eye_img, hp_outputs):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        try:
            self.exec_network.start_async(request_id=0, inputs={'left_eye_image': l_eye_img,
                                                                'right_eye_image': r_eye_img,
                                                                'head_pose_angles': hp_outputs})
        except Exception as e:
            log.error("Following Error occurred during inference request for gaze-estimation: " + str(e), exc_info=True)
        return


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        # Wait for the async request to be complete
        # -1 means wait until task is completed then return status
        status = self.exec_network.requests[0].wait(-1)
        return status


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        try:
            output = self.exec_network.requests[0].outputs
        except Exception as e:
            log.error("Following error occurred while returning output for gaze-estimation model: " + str(e),
                      exc_info=True)
        return output
