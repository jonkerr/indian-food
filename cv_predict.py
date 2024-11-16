# Force CPU only for predictions
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import cv2
import numpy as np
from image_prep import get_label_mapping, IMG_SIZE

class TastyFoodPredictor():
    def __init__(self, model_path = 'models/efficientnet_v2_20_84.64.keras', preprocess_input_fn=None):
        self.softmax = keras.layers.Softmax()
        self.model = keras.models.load_model(model_path)
        self.mapping_dct = get_label_mapping()
        
        self.preprocess_input = preprocess_input_fn
        if preprocess_input_fn is None:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            self.preprocess_input = preprocess_input
            
        # pre-warm
        self.predict('data/Food_Classification/chole_bhature/002.jpg')
        
        
    def predict(self, image_path):
        # pre-process the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = np.reshape(img, [1,IMG_SIZE[0], IMG_SIZE[1], 3])
        img = np.array(img, dtype=np.float32)
        img = self.preprocess_input(img)
        
        # predict
        preds_out = self.model.predict(img) 

        # map to label
        idx = np.argmax(preds_out[0])
        return self.mapping_dct[idx]
    
    
def health_check():       
    print('starting health check')
    test_image = 'data/Food_Classification/chole_bhature/002.jpg'
    # confirm key files exist
    import os
    import sys
    
    if os.path.exists(test_image):
        print('found test image')
    else:
        print('error importing image')
        sys.exit()
        
    model = TastyFoodPredictor()
    pred = model.predict(test_image)
    print(f'Predicted: {pred}')
    

if __name__ == "__main__":
    health_check()