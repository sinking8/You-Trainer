from PIL import Image
from keras.preprocessing import image

import tensorflow as tf
import numpy as np

class Model:
    class_names = ["Correct","Not Correct"]
    batch_size = 32
    img_height = 24
    img_width = 24
    def __init__(self,model_dir="./model/model.h5"):
        self.model_path = model_dir
        self.model = tf.keras.models.load_model(model_dir)
        
    def predictions(self,img_dir):
        img = image.load_img(
            img_dir, target_size=(self.img_height, self.img_width)
        )
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        return {"text":"{} Posture"
                .format(self.class_names[np.argmax(score)])}