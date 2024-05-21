import numpy as np
# import matplotlib.pyplot as plt
import cv2
import os

print('importing tensorflow')
from tensorflow.keras.models import load_model
print('finished importing tensorflow')

class IntersectionManager:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def crop_image(self,image):
        height, width = image.shape[:2]
    
        # Izračunavanje koordinata
        x1 = width // 3 - 40
        x2 = width * 2 // 3 + 30 + 20
        y1 = height // 3 + 40 + 20+10
        y2 = height * 2 // 3 + 30 + 16
        cropped_image = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped_image,cv2.COLOR_RGB2GRAY)
        gray = gray/255
        return gray

    def classify(self,frame):
        img = self.crop_image(frame)
        # img = img / 255
        img = img.reshape(1,96, 261, 1)
        predictions = self.model.predict(img, verbose=0) # niz verovatnoca [p1,p2,p3,p4] ako hoces verovatnoce samo ovo vrati
        predicted_class = np.argmax(predictions, axis=-1)[0] # uzima najverovatniju
        return predicted_class

if __name__ == '__main__':
    #----------------------------------------------------------
    test_images = os.listdir('RaskrsnicaTest')
    test_image_path = os.path.join('RaskrsnicaTest', test_images[1]) 
    test_image = cv2.imread(test_image_path)
    print(type(test_image))

    #Ako zelis da prikazes koju sliku si izabrao
    plt.imshow(test_image)
    plt.axis("off")
    plt.show()
    #----------------------------------------------------------
    
    # Uokvireni kod nece postojati za raspberry pi

    c = IntersectionManager('models/raskrsnica_95%_.h5') 
    print(c.classify(test_image))
