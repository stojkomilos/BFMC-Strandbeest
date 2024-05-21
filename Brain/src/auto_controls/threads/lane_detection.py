import cv2
import numpy as np
import time
import globals

# great explanation for all of this code for lane navigation: https://www.youtube.com/watch?v=eLTLtUVuuy4

class LineException(Exception):
    pass

global USE_YT_LANE_DETECTION_PARAMETERS

class LaneDetector():
    def __init__(self):
       if globals.MEASURE_SIGN_DETECTION_TIME:
        self.i = 0
        self.cum_time = 0

    def display_lines(self, image, lines):

        line_image = np.zeros_like(image)

        if lines is not None:
            for x1, y1, x2, y2  in lines:
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    if globals.LANE_DEBUG_PRINT:
                        print("===== Detected INVALID LINE, ignoring it: ", (x1, y1), (x2, y2))
                    continue
                try:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 15)
                except Exception as e:
                    if globals.LANE_DEBUG_PRINT:
                        print("===== Detected INVALID LINE, ignoring it: ", (x1, y1), (x2, y2))
                        print("================== LINE FATAR ERROR, IGNORING ERROR THIS IS BAD PRACTISE")

        return line_image

    def do_edge_detection(self, image, head):

        # returns a transformed image: canny(blur(gray( image )))
        # TEMP, this exists because current video is not grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0) # applied to reduce noise

        if globals.USE_YT_LANE_DETECTION_PARAMETERS:
            canny = cv2.Canny(blurred,50,150)
        else:
            canny = cv2.Canny(blurred, 180, 255) # the actual edge detection part

        # head.set_debug_image(canny)

        return canny

    def stojko_select_region_of_interest(self, image, head, frame):

        # use to visualise and find coordinates of /bottom_left/, /bottom_right/ and /top
        #plt.imshow(image)
        #plt.show()

        # this needs to be manually adjusted for every /image/ resolution, as it is RESOLUTION DEPENDENT
        height = image.shape[0]
        width = image.shape[1]
        
        k1 = 720.0/360
        k2 = 1280.0/512

        bottom_left = (75*k1, height)
        bottom_right = (470*k1,height)
        top = (267*k1,160*k2)
        A = (0, 260*k2)
        B = (0, height)
        C = (140*k1, height)
        D = (180*k1, 275*k2)
        E = (340*k1, 275*k2)
        F = (390*k1, height)
        G = (width, height)
        H = (width, 260*k2)
        I = (340*k1, 156*k2)
        J = (200*k1, 160*k2)

        temp = [(int(i[0]), int(i[1])) for i in [A, B, C, D, E, F, G, H, I, J]]
        polygons = np.array([temp]) # RESOLUTION DEPENDENT

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255) # sets our region of interest to the color /white/ in our "mask"

        masked_image = cv2.bitwise_and(image, mask)

        #head.set_debug_image(masked_image)

        return masked_image
    

    def select_region_of_interest_very_cropped(self, image, head, frame):
        # use to visualise and find coordinates of /bottom_left/, /bottom_right/ and /top
        #plt.imshow(image)
        #plt.show()

        # this needs to be manually adjusted for every /image/ resolution, as it is RESOLUTION DEPENDENT
        height = image.shape[0]
        width = image.shape[1]
        
        bottom_left = (75, height)
        bottom_right = (470,height)
        top = (267,160)
        A = (0, 230)
        B = (0, 240)
        C = (width, 240)
        D = (width, 230)
        # E = (340, 200)
        # F = (390, 200)
        # G = (width, 200)
        # H = (width, 200)
        # I = (340, 156)
        # J = (200, 160)
        polygons = np.array([[A, B, C, D]] ) # RESOLUTION DEPENDENT

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255) # sets our region of interest to the color /white/ in our "mask"

        masked_image = cv2.bitwise_and(image, mask)

        # for debug
        # grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # head.debug_image = cv2.bitwise_and(grayscale, mask)

        return masked_image
    

    def select_region_of_interest(self, image, head, frame):

        # use to visualise and find coordinates of /bottom_left/, /bottom_right/ and /top
        #plt.imshow(image)
        #plt.show()

        # this needs to be manually adjusted for every /image/ resolution, as it is RESOLUTION DEPENDENT
        height = image.shape[0]
        width = image.shape[1]
        
        bottom_left = (75, height)
        bottom_right = (470,height)
        top = (267,160)
        A = (0, 260)
        B = (0, height)
        C = (140, height)
        D = (180, 275)
        E = (340, 275)
        F = (390, height)
        G = (width, height)
        H = (width, 260)
        I = (340, 156)
        J = (200, 160)
        polygons = np.array([[A, B, C, D, E, F, G, H, I, J]] ) # RESOLUTION DEPENDENT


        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255) # sets our region of interest to the color /white/ in our "mask"

        masked_image = cv2.bitwise_and(image, mask)

        # for debug
        # grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # head.debug_image = cv2.bitwise_and(grayscale, mask)


        # use this to visualise the result of the bitmask
        # cv2.imshow("result", masked_image)
        # cv2.waitKey(0)

        return masked_image
    
    # main function for detecting the lanes
    def get_lane_features(self, frame, large_frame, head):
        # returned display_helper is None if no lines are detected
        # frame_features['average_line_XXX] is None if there are no XXX lines detected, where XXX is either left or right

        if globals.MEASURE_LANE_DETECTION_TIME:        
            self.last_time = time.time()

        # if globals.USE_YT_LANE_DETECTION_PARAMETERS:
        #     frame = cv2.resize(frame, (1280, 720))

        if globals.MEASURE_SIGN_DETECTION_TIME:
            self.last_time = time.time()

        edge_img  = self.do_edge_detection(large_frame, head)

        if globals.USE_YT_LANE_DETECTION_PARAMETERS:
            cropped_img = self.stojko_select_region_of_interest(edge_img, head, large_frame)
        else:
            cropped_img = self.select_region_of_interest(edge_img, head, large_frame)

        # head.set_debug_image(cropped_img)

        if globals.USE_YT_LANE_DETECTION_PARAMETERS:
            lines = cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength = 40,maxLineGap = 5)
        else:
            assert False # deprecated, stop using
            # arguments that might have to be changed with resolution: 2nd, 4th, minLineLength, maxLineGap. Refer to around 53:00 https://www.youtube.com/watch?v=eLTLtUVuuy4
            lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 40, np.array([]), minLineLength=25, maxLineGap=10) # BIG PART OF LINE IS RESOLUTION DEPENDENT

        average_line_left, average_line_right, display_helper = self.lines_to_left_and_right_line(large_frame, lines)
            
        frame_features = dict()
        frame_features['average_line_left'] = average_line_left
        frame_features['average_line_right'] = average_line_right

        if display_helper is not None:
            display_helper = self.display_lines(large_frame, display_helper)

        if globals.MEASURE_SIGN_DETECTION_TIME:
            self.i += 1
            self.cum_time += time.time() - self.last_time
            
            if self.i == 10:
                avg_time = self.cum_time/10
                self.i = 0
                self.cum_time = 0
                print('get_lane_features() (lane detection) average time', avg_time, 's')
        
        # rescale so that PID can work with it from the old tuning before using the new resolution
        #if globals.USE_YT_LANE_DETECTION_PARAMETERS and frame_features['average_line_left'] is not None and frame_features['average_line_right'] is not None:
        #    for key in frame_features:
        #        frame_features[key][0] = frame_features[key][0] * (512.0/1280) # rescale the slope
        #        frame_features[key][1] = frame_features[key][1] * (360/720.0) # rescale the y intercept
            
        return frame_features, display_helper

    def lines_to_left_and_right_line(self, image, lines):
        # first return value is None if there are no left lines detected
        # second return value is None if there are no right lines detected
        # third return value is None if there are no lines detected at all (neither left nor right)

        if lines is None:
            return None, None, None

        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            slope, y_intercept = np.polyfit((x1, x2),(y1, y2), 1) # if our line is: y=a*x+b , then: a=slope, b=y_intercept

            threshold = 0
            assert threshold >= 0

            if slope < -threshold:
                left_lines.append((slope, y_intercept))
            elif slope > threshold:
                right_lines.append((slope, y_intercept))

        left_line_exists = len(left_lines) > 0
        right_line_exists = len(right_lines) > 0

        left_lines_average = np.average(left_lines, axis = 0) if left_line_exists else None
        right_lines_average = np.average(right_lines, axis = 0) if right_line_exists else None

        if not left_line_exists and not right_line_exists:
            return None, None, None

        left_line = self.make_coordinates(image, left_lines_average[0], left_lines_average[1]) if left_line_exists else None
        right_line = self.make_coordinates(image, right_lines_average[0], right_lines_average[1]) if right_line_exists else None

        if left_line_exists and right_line_exists:
            display_helper = np.array([left_line, right_line])
        elif left_line_exists and not right_line_exists:
            display_helper = np.array([left_line])
        elif not left_line_exists and right_line_exists:
            display_helper = np.array([right_line])
        else:
            display_helper = None

        return left_lines_average, right_lines_average, display_helper
    
    def horizontal_line_detection(self, frame, head):
        edge_img = self.do_edge_detection(frame, head)
        # cropped_img = self.select_region_of_interest_horiz(edge_img, head, frame)
        cropped_img = self.select_region_of_interest_very_cropped(edge_img, head, frame)
        lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 40, np.array([]), minLineLength=25, maxLineGap=10) # BIG PART OF LINE IS RESOLUTION DEPENDENT

        if lines is None:
            return False
        
        horiz_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            k, n = np.polyfit((x1, x2), (y1, y2), 1)

            if abs(k) < 0.2:
                horiz_lines.append((k, n))
        
        return len(horiz_lines) > 0
        

    def make_coordinates(self, image, slope, y_intercept):

        y1 = image.shape[0] # y1 = lowest point
        y2 = int(y1*(3/5)) # y2 = highest point, 3/5 is an abritrary choise for drawing pretty lines. Refer to 1:15:00 - https://www.youtube.com/watch?v=eLTLtUVuuy4&t=3343s

        x1 = int((y1-y_intercept)/slope)
        x2 = int((y2-y_intercept)/slope)

        return np.array([x1,y1,x2,y2])


    # # ideja: canny AND decolorize(frame) se gura u houghTransform
    def decolorize(self, image):
        # returns a boolean mask of the same shape as the input image, where True indicates that the pixel is colored

        grayness = (image[:,:,0]+image[:,:,1],image[:,:,2])/3
        colorness = (np.abs(image[:,:,0]-grayness)+np.abs(image[:,:,1]-grayness)+np.abs(image[:,:,2]-grayness))/3
        is_colored = (colorness > grayness/5)

        return is_colored
