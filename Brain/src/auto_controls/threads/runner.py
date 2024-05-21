import cv2
import numpy as np
import time
import base64

import sys
sys.path.append("src/auto_controls/threads")
import globals
from head import Head

class Runner():

    def __init__(self):
        if not globals.IS_ONLINE:
            # self.video = cv2.VideoCapture("youtube_tutorial.mp4")
            self.video = cv2.VideoCapture("2.avi")

        self.thinker = Head()
        self.engine_run = False

        print("IS_ONLINE=", globals.IS_ONLINE)
    
    def do_iteration_offline(self):
        _, frame = self.video.read()

        self.thinker.do_iteration(frame)
        angle, speed, disp_image = self.thinker.current_steering_angle, self.thinker.speed, self.thinker.get_debug_image()

        # keep this. Seperates each frame to make output prettier
        print(' ')

        cv2.imshow('result', self.thinker.get_debug_image())
        cv2.waitKey(1)

        # to slow down the video
        time.sleep(0.05) # 0.2 was chosen arbitrarily, it can be changed

    def do_iteration_online(self, parent_object):

        c_valid, c_msg_value, _,_,_ = parent_object.get_camera_msg()

        if not c_valid:
            return
            
        # viktor
        engine_run = parent_object.get_engine_run_msg()
        if(engine_run is not None):
            if(self.engine_run != engine_run):
                self.thinker.reset_pid()
                self.engine_run = engine_run
        
        decoded_bytes = base64.b64decode(c_msg_value)
        image_arr = np.fromstring(
            decoded_bytes, np.uint8
        )  # Convert bytes to numpy array
        np_frame = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

        # Process the data
        self.thinker.do_iteration(np_frame)
        angle, speed, disp_image = self.thinker.current_steering_angle, self.thinker.speed, self.thinker.get_debug_image()

        # keep this. Seperates each frame to make output prettier
        print(' ')

        if disp_image is not None:
            _, encoded_img = cv2.imencode(".jpg", disp_image)
            image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
        else:
            assert(disp_image is None)
            image_data_encoded = None

        parent_object.send(angle, speed, image_data_encoded) 

    def run(self):
        if globals.IS_ONLINE:
            self.run_online()
        else:
            self.run_offline()

if globals.IS_ONLINE is False:
    offline_runner = Runner()
    while offline_runner.video.isOpened():
        offline_runner.do_iteration_offline()
    offline_runner.video.release()
    cv2.destroyAllWindows()
    
# This is how you run it online in threadAutoControl.py
# def run(self):
#     while self._running:
#           self.do_iteration_online()