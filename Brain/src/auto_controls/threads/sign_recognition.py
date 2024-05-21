import numpy as np
# import matplotlib.pyplot as plt
import cv2
import os
import time

import globals

if globals.IS_ONLINE:
    print("starting importing tf lite")
    import tflite_runtime.interpreter as tflite
    print("done importing tf lite")

class Classifier:

    def __init__(self, model_path):
        if globals.IS_ONLINE is False:
            return

        # Load TensorFlow Lite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        # Get the index of input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']
        
        if globals.MEASURE_SIGN_DETECTION_TIME:
            self.i = 0
            self.cum_time = 0
        
    def crop_right_third(self, image):
        one_third_width = image.shape[1] // 3
        startX = 2 * one_third_width - 150  # Begin at the right third
        endX = image.shape[1]  # End of the image (right edge)
        startY = image.shape[0] // 3 - 100  # From the top of the image
        endY = image.shape[0] - 190  # To the bottom of the image
        cropped_image = image[startY:endY, startX:endX]
        # cropped_image = cv2.resize(cropped_image, (120, 120), interpolation=cv2.INTER_AREA)
        cropped_image = cv2.resize(cropped_image, (150, 150), interpolation=cv2.INTER_AREA)
        return cropped_image

    def classify(self, frame):

        if globals.IS_ONLINE is False:
            return 1 # return empty if run offline

        if frame is None:
            return -1  # Ensure frame is not None

        if globals.MEASURE_SIGN_DETECTION_TIME:
            self.last_time = time.time()

        img = self.crop_right_third(frame)
        img = img / 255.0  # Normalization
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)  # Adding batch dimension

        # Setting the input tensor
        self.interpreter.set_tensor(self.input_index, img)
        self.interpreter.invoke()

        # Getting the results
        predictions = self.interpreter.get_tensor(self.output_index)[0]
        max_prediction = np.amax(predictions)

        if max_prediction < 0.7:
            result = 1  # Assuming "empty"
        else:
            result = np.argmax(predictions)

        if globals.MEASURE_SIGN_DETECTION_TIME:
            self.i += 1
            self.cum_time += time.time() - self.last_time
            if self.i == 10:
                avg_time = self.cum_time / 10
                self.i = 0
                self.cum_time = 0
                print('Sign recognition average time:', avg_time, 's')

        return result

    def ordinal_to_category(self, ordinal):
        if ordinal == 0:
            return "pesak"
        elif ordinal == 1:
            return "prazno"
        elif ordinal == 2:
            return "prednost"
        elif ordinal == 3:
            return "stop"
        elif ordinal == 4:
            return "parking"
        elif ordinal == 5:
            return "autoput izlaz"
        elif ordinal == 6:
            return "autoput ulaz"
        elif ordinal == 7:
            return "desno"
        elif ordinal == 8:
            return "kruzni tok"
        elif ordinal == 9:
            return "zabranjen ulaz"
        else:
            raise Exception("Ordinal out of bounds")

if __name__ == '__main__':
    # for testing, not to be run on the car

    test_images = os.listdir('PesakTest2')
    test_image_path = os.path.join('PesakTest2', test_images[1]) 
    test_image = cv2.imread(test_image_path)
    print(type(test_image))

    c = Classifier('1_100%test_cnn.h5') 
    print(c.classify(test_image))