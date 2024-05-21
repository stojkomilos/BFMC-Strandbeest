import time
import globals

import cv2

class Pid():
    def __init__(self):
        self.i = 0
        self.prev_err = 0
        self.err_integral = 0
        self.prev_time = time.time()

    def do_iteration(self, frame_features, frame, large_frame):

        dt = time.time() - self.prev_time

        left_slope, left_y_intercept = frame_features['average_line_left'] 
        left_x_intercept = ( large_frame.shape[0] - left_y_intercept )/left_slope

        right_slope, right_y_intercept = frame_features['average_line_right'] 
        right_x_intercept = ( large_frame.shape[0] - right_y_intercept )/right_slope

        current = (left_x_intercept+right_x_intercept)/2
        err = current - large_frame.shape[1]/2 

        err = err * (512.0/1280) # rescale to use Viktor's PID tuning
        

        if globals.PID_DEBUG_PRINT:
            print(f"{left_x_intercept=}, {right_x_intercept=}")
            print()

        # self.err_list.append(err)
        # self.err_integral += err*dt

        # TODO: ograniciti
        # ko god bude ove koeficijente menjao, mozda probati da se izbegne "derivative" koeficijent tj da se stavi na 0, tj ostaviti k_der=0
        # k_prop, k_int, k_der = 0.01, 0.000, 0
        #k_prop, k_int, k_der = 0.35, 0.07, 0
        k_prop, k_int, k_der = 0.35, 0.07, 0
        steer_angle = self.calc_pid(self.prev_err, err, self.err_integral, dt, k_prop, k_int, k_der, self.i)

        if globals.PID_DEBUG_PRINT:
            print(f'{err=} {steer_angle=} {self.err_integral=}')
        
        if steer_angle >= 23:
            steer_angle = 23
        elif steer_angle <= -23:
            steer_angle = -23
        
        if abs(steer_angle) != 23:
            self.err_integral += (err + self.prev_err)/2*dt

        self.prev_time = time.time()
        self.prev_err = err

        if globals.PID_DEBUG_PRINT:
            print('pid output: ', steer_angle)

        return steer_angle


    def calc_pid(self, prev_err, err, err_integral, dt, k_prop, k_int, k_der, i):
        if i == 0:
            derivative_term = 0
        else:
            derivative_term = k_der*((err - self.prev_err) / dt)

        if globals.PID_DEBUG_PRINT:
            print('proportional:', k_prop*err, ' integral:', k_int*err_integral, ' derivatrive: ', derivative_term)

        return k_prop*err + k_int*err_integral + derivative_term