import cv2
import numpy as np
import time

from enum import Enum
from termcolor import colored
from collections import defaultdict 

assert __name__ != "__main__" # you should run "runner.py" instead of this

import globals
from pid import Pid

if globals.IS_ONLINE:
    if globals.DO_SIGN_DETECTION:
        from src.auto_controls.threads.sign_recognition import Classifier
    from src.auto_controls.threads.lane_detection import LaneDetector
    from src.auto_controls.threads.pedestrian_manager import PedestrianManager

    if globals.DO_SEMAPHORE_DETECTION:
        from src.auto_controls.threads.semaphore_manager import SemaphoreManager
else:
    if globals.DO_SIGN_DETECTION:
        from sign_recognition import Classifier
    from lane_detection import LaneDetector
    from pedestrian_manager import PedestrianManager

    if globals.DO_SEMAPHORE_DETECTION:
        from semaphore_manager import SemaphoreManager

class Turn(Enum):
    TURN_LEFT = -1
    TURN_FORWARD = 0
    TURN_RIGHT = 1

class Head():

    def __init__(self):
        self.pid = Pid()
        # self.nr_frame = 0
        self.lane_detector = LaneDetector()

        self.CONST_SPEED = -25
        self.STOP_WAIT = 1 # how much the car waits on stop sign in seconds
        self.INTERSECTION_TURN_ANGLE = -18
        self.MAX_TURN_ANGLE = 20
        self.TURN_LENGTH_CONSTANT = 175 
        self.CROSSWALK_SLOWDOWN = 6
        self.SIGN_COOLDOWN = 10

        self.parking_state = 1
        self.sign_classification_history = []
        self.intersection_classification_history = []
        self.do_this_frame = True 
        self.prev_steering_angle = 0
        self.right_not_detected_counter = 0
        self.left_not_detected_counter = 0
        self.start_time = time.time()

        self.lane_navigation_i = 0
        self.lane_navigation_cum_time = 0

        self.pedestrian_history = []
        self.pedestrian_manager = PedestrianManager('/home/milos/BFMC-Strandbeest/Brain/src/auto_controls/threads/models/Pedestrian_RO1.tflite')

        if globals.DO_SEMAPHORE_DETECTION:
            self.semaphore_history = []
            self.semaphore_manager = PedestrianManager('/home/milos/BFMC-Strandbeest/Brain/src/auto_controls/threads/models/Semaphore.tflite')

        if globals.DO_SIGN_DETECTION:
            import sys
            # sys.path.append("src/auto_controls/threads")
            self.classifier = Classifier('/home/milos/BFMC-Strandbeest/Brain/src/auto_controls/threads/models/MobileNetV1_V14_97__150_150_G_half_RO1.tflite')
        
        self.state = globals.State.LANE_FOLLOWING
        self.is_debug_image_set = True
        self.current_steering_angle = 0

        if globals.DISPLAY_LANE_DETECTION_DEBUG:
            assert globals.SEND_DEBUG_IMAGE # it makes no sense to have debug display for lane detection if you're not even sending debug image
        
        self.frame_counter = 0
        self.frame_time_arr = []
        self.frame_time_cum = 0

    def check_similarity(self, history, window_length, threshold):
        assert len(history) >= window_length
        assert threshold >= window_length/2
        
        d = defaultdict(lambda : 0)
        for i in range(len(history)-1, len(history)-1-window_length, -1):
            d[history[i]] = d[history[i]] + 1
            if d[history[i]] >= threshold:
                return history[i]

        return None
            

    def reset_pid(self):
        self.pid = Pid()

    def set_debug_image(self, image):
        # if this fails, it means that the debug image is being set multiple 
        # times in one iteration, which is bad practise
        assert self.is_debug_image_set is False
        assert image is not None
        self.is_debug_image_set = True
        self.__debug_image = image
        assert globals.SEND_DEBUG_IMAGE is True
    
    def get_debug_image(self):
        return self.__debug_image
    
    def set_steering_angle(self, angle):
        self.prev_steering_angle = float(self.current_steering_angle)
        self.current_steering_angle = float(angle)

    def set_speed(self, speed):
        self.speed = float(speed)
        
    # main function for lane following
    def do_lane_navigation(self, lane_features, frame, large_frame):

        baba = 20

        if globals.MEASURE_LANE_NAVIGATION_TIME:        
            self.lane_navigation_start_time = time.time()

        if lane_features['average_line_left'] is not None and lane_features['average_line_right'] is not None:
            self.right_not_detected_counter = 0
            self.left_not_detected_counter = 0

            self.set_steering_angle(self.pid.do_iteration(lane_features, frame, large_frame))
            self.set_speed(self.CONST_SPEED)

        elif lane_features['average_line_left'] is not None and lane_features['average_line_right'] is None:
            if globals.LANE_DEBUG_PRINT:
                print(colored("=== RIGHT LANE NOT DETECED", 'magenta'))

            self.right_not_detected_counter += 1
            # angle = self.prev_steering_angle + self.right_not_detected_counter**2
            angle = baba

            self.set_steering_angle(angle)
            self.set_speed(self.CONST_SPEED)

        elif lane_features['average_line_left'] is None and lane_features['average_line_right'] is not None:
            if globals.LANE_DEBUG_PRINT:
                print(colored("=== LEFT LANE NOT DETECED", 'magenta'))
            
            self.left_not_detected_counter += 1
            # angle = self.prev_steering_angle - self.left_not_detected_counter**2  
            angle = -baba

            self.set_steering_angle(angle)
            self.set_speed(self.CONST_SPEED)

        elif lane_features['average_line_left'] is None and lane_features['average_line_right'] is None:
            if globals.LANE_DEBUG_PRINT:
                print(colored("=== NO LANE DETECTED", 'magenta'))

            self.set_steering_angle(self.prev_steering_angle) # if there are no lanes, keep the previous angle
            self.set_speed(self.CONST_SPEED/3) # slow down to reduce the image blur and maybe that will help detect the lanes
        else:
            assert False

        if globals.MEASURE_LANE_NAVIGATION_TIME:
            self.lane_navigation_i += 1
            self.lane_navigation_cum_time += time.time() - self.lane_navigation_start_time
            
            if self.lane_navigation_i == 10:
                avg_time = self.lane_navigation_cum_time/10
                self.lane_navigation_i = 0
                self.lane_navigaiton_cum_time = 0
                print('do_lane_navigation() (lane navigation) average time', avg_time, 's')

    # This is the main function of the project that is run once every frame
    def do_iteration(self, frame):

        OFFSET = 10
        self.frame_counter += 1
        self.frame_time_arr.append(time.time())

        if self.frame_counter % 10 == 0:
            print('average TOTAL frame time: ', (self.frame_time_arr[-1] - self.frame_time_arr[-OFFSET])/OFFSET)
        
        if self.frame_counter % 10 == 0:
            globals.SEND_DEBUG_IMAGE = True
            globals.DISPLAY_LANE_DETECTION_DEBUG = False
        else:
            globals.SEND_DEBUG_IMAGE = False
            globals.DISPLAY_LANE_DETECTION_DEBUG = False


        if globals.SEND_DEBUG_IMAGE:
            # sets a default debug frame
            self.is_debug_image_set = False
            # if globals.DISPLAY_DEBUG:
            self.set_debug_image(frame)
            self.is_debug_image_set = False
        else:
            self.__debug_image = None

        large_frame = cv2.resize(frame, (1280, 720))

        lane_features, lane_display_helper = self.lane_detector.get_lane_features(frame, large_frame, self)

        # show image with overlay
        if globals.SEND_DEBUG_IMAGE and globals.DISPLAY_LANE_DETECTION_DEBUG is True and self.get_debug_image() is not None and lane_display_helper is not None:

            # SPORO
            if globals.USE_YT_LANE_DETECTION_PARAMETERS:
                lane_display_helper = cv2.resize(lane_display_helper, (512, 360)) # resize so it can be displayed and lanes stuff added nicely 

            # assert False

            image = cv2.addWeighted(self.get_debug_image(), 0.8, lane_display_helper, 1, 1)
            self.set_debug_image(image)
        
        if globals.DO_PEDESTRIAN:

            # all the pedestrian processing
            cur_pedestrian = self.pedestrian_manager.classify(frame)
            if cur_pedestrian == 1:
                print("PEDESTRIAN NEAR DETECTED")
            if cur_pedestrian == 2:
                print("PEDESTRIAN FAR DETECTED")

            self.pedestrian_history.append(cur_pedestrian)
            if self.pedestrian_manager.check_above_threshold(self.pedestrian_history, window=2, threshold=2):
                self.set_speed(0)
                return

        if globals.DO_SEMAPHORE_DETECTION:
            # all the semaphore processing
            cur_semaphore = self.semaphore_manager.classify(frame)
            self.semaphore_history.append(cur_semaphore)
            if self.semaphore_manager.check_above_threshold(self.semaphore_history, window=3, threshold=2):
                self.set_speed(0)
                return

        match self.state:

            case globals.State.LANE_FOLLOWING:

                self.do_lane_navigation(lane_features, frame, large_frame)

                if globals.DO_SIGN_DETECTION: 
                    self.do_sign_detection(frame)

            case globals.State.GO_TO_STOP_LINE:
                self.do_lane_navigation(lane_features, frame)
                detected_horiz_lane = self.lane_detector.horizontal_line_detection(frame, self)
                if detected_horiz_lane:
                    print("detected horizontal lane !")
                    self.state = self.next_state
                    self.start_time = time.time()
                return

            case globals.State.INTERSECTION:
                # turn_time = self.TURN_LENGTH_CONSTANT / -self.CONST_SPEED
                turn_time = 9
                # print(turn_time)

                if time.time() - self.start_time < turn_time:
                    if time.time() - self.start_time < 6:
                        self.set_steering_angle(3)
                        self.set_speed(self.CONST_SPEED)
                    else:
                        self.set_steering_angle(self.INTERSECTION_TURN_ANGLE)
                        self.set_speed(self.CONST_SPEED)
                    return

                else:
                    self.state = globals.State.LANE_FOLLOWING

                    self.set_steering_angle(self.prev_steering_angle)
                    self.set_speed(self.CONST_SPEED)
                    return
            
            case globals.State.WAITING:
                if time.time() - self.start_time < self.STOP_WAIT:

                    self.set_steering_angle(0)
                    self.set_speed(0)
                    return

                else:
                    self.state = globals.State.INTERSECTION
                    self.start_time = time.time()
                    self.set_steering_angle(0)
                    self.set_speed(self.CONST_SPEED)
                    return

            case globals.State.SLOW_DOWN: #Dodato
                if time.time() - self.start_time <  self.CROSSWALK_SLOWDOWN:
                    self.do_lane_navigation(lane_features, frame)
                    self.set_speed(-17)
                    return
                else:
                    self.state = globals.State.LANE_FOLLOWING
                    self.start_time = time.time()
                    self.set_speed(self.CONST_SPEED)
                    return
                
            case globals.State.PARKING:
                match self.parking_state:
                    case 1:
                        self.do_lane_navigation(lane_features, frame)
                        if time.time() - self.start_time > 1:
                            self.parking_state = 2
                            self.start_time = time.time()
                        return
                    case 2:
                        if time.time() - self.start_time < 2:
                            self.set_speed(-20)
                            self.set_steering_angle(-20)
                        else:
                            self.parking_state = 3
                            self.start_time = time.time()
                    case 3:
                        if time.time() - self.start_time < 2:
                            self.set_speed(0)
                            self.set_steering_angle(0)
                        else:
                            self.parking_state = 4
                            self.start_time = time.time()
                    case 4:
                        if time.time() - self.start_time < 2:
                            self.set_speed(self.CONST_SPEED)
                            self.set_steering_angle(20)
                        else:
                            self.parking_state = 1
                            self.start_time = time.time()
                            self.state = globals.State.LANE_FOLLOWING

            case _:
                assert False
    
    def do_sign_detection(self, frame):
        sign = self.classifier.classify(frame)

        self.sign_classification_history.append(sign)
        # debug printing

        NO_SIGN = 1
        if sign != NO_SIGN:
            print("Detected sign in SINGLE frame:", self.classifier.ordinal_to_category(sign))


        WINDOW_LENGTH = 2
        threshold = 2

        if len(self.sign_classification_history) >= WINDOW_LENGTH:
            history_sign = self.check_similarity(self.sign_classification_history, WINDOW_LENGTH, threshold)

            if history_sign is not None and history_sign != NO_SIGN:
                print_str = "DEFINITE SIGN DETECTION:", self.classifier.ordinal_to_category(history_sign)
                print(colored(print_str, 'red'))

                SIGN_DEBUG = False               
                # if SIGN_DEBUG is False:
                #     # actually stop on stop sign
                #     if self.classifier.ordinal_to_category(sign) == "stop":
                #         self.next_state = globals.State.WAITING
                #         self.state = globals.State.GO_TO_STOP_LINE
                #         self.start_time = time.time()

                #         # self.set_speed(0)
                #         # self.set_steering_angle(0)
                #         return
                    
                #     if self.classifier.ordinal_to_category(sign) == "prednost":
                #         self.next_state = globals.State.INTERSECTION
                #         self.state = globals.State.GO_TO_STOP_LINE
                #         self.start_time = time.time()

                #         # self.set_speed(0)
                #         # self.set_steering_angle(0)
                #         return
                    
                
                #     if self.classifier.ordinal_to_category(sign) == "pesak":
                #         self.next_state = globals.State.SLOW_DOWN
                #         self.state = globals.State.GO_TO_STOP_LINE
                #         self.start_time = time.time()
                        
                #         # self.set_speed(self.speed*2/3)
                #         return
                    
                    # self.set_speed(self.speed*2/3)
                return