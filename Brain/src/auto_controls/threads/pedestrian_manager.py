import numpy as np
import cv2
import time

import globals

if globals.IS_ONLINE:
    print("starting importing tf lite")
    import tflite_runtime.interpreter as tflite
    print("done importing tf lite")



class PedestrianManager:
    def check_above_threshold(self, history, window=None, threshold=None):

        assert window is not None and threshold is not None

        CLASS_NEAR = 1
        CLASS_FAR = 1
        count = 0

        for i in history[len(history)-1-window:]:
            if i == CLASS_NEAR or i == CLASS_FAR:
                count += 1
        
        if count >= threshold:
            return True
        else:
            return False

    def __init__(self, tflite_model_path):
        # Učitavanje TensorFlow Lite modela
        if globals.IS_ONLINE is False:
            return

        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        # # Dobijanje indeksa ulaznih i izlaznih tenzora
        # self.input_details = self.interpreter.get_input_details()
        # self.output_details = self.interpreter.get_output_details()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']
        self.i = 0
        self.cum_time = 0

    def crop_right_third(self,image):
        one_third_width = image.shape[1] // 3
        startX = one_third_width - 30
        endX = image.shape[1]  # Kraj slike (desni rub)
        startY = image.shape[0]//3 -60# Od vrha slike
        endY = 2*image.shape[0]//3 # Do dna slike
        cropped_image = image[startY:endY, startX:endX]
        cropped_image = cv2.resize(cropped_image, (150, 150), interpolation=cv2.INTER_AREA)

        return cropped_image
    
    def classify(self, frame):

        if globals.IS_ONLINE is False:
            return 0

        if frame is None:
            return -1  # Ensure frame is not None

        if globals.MEASURE_PEDESTRIAN_DETECTION_TIME:
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

        if max_prediction < 0.5:
            result = 0  
        else:
            result = np.argmax(predictions)

        if globals.MEASURE_PEDESTRIAN_DETECTION_TIME:
            self.i += 1
            self.cum_time += time.time() - self.last_time
            if self.i == 10:
                avg_time = self.cum_time / 10
                self.i = 0
                self.cum_time = 0
                print('Pedestrian recognition average time:', avg_time, 's')

        return result
    def ordinal_to_category(self, ordinal):
        if ordinal == 0:
            return "Nema pesaka"
        elif ordinal == 1:
            return "Blizu pesak"
        elif ordinal == 2:
            return "Daleko pesak"
        else:
            raise Exception("Ordinal out of bounds")
    


if __name__ == '__main__':
    classifier = PedestrianManager('Pedestrian.tflite')
    #print(test_speed(classifier))

    # 0 -----> Nema
    # 1 -----> Blizu
    # 2 -----> Daleko