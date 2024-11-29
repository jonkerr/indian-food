# Force CPU only for predictions
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import cv2
import numpy as np
from image_prep import get_label_mapping, IMG_SIZE
from cv_model import get_empty_model

# standardize on this for now
import tensorflow as tf

class TastyFoodPredictor():
    def __init__(self, model_path = 'models/efficientnet_v2_20_84.64.keras', preprocess_input_fn=None, 
                 num_classes=20, weights_path = 'models/weights/efficientnet_v2_20_84.64.hdf5', lite_model=True, tflite_path='models/lite/efficientnet_v2_20_84.64.tflite'):
#        self.softmax = keras.layers.Softmax()

        # load the whole shebang
        #self.model = keras.models.load_model(model_path)
                
        self.mapping_dct = get_label_mapping()
        
        self.preprocess_input = preprocess_input_fn
        if preprocess_input_fn is None:
            self.preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
            
            
        self.lite_model = lite_model
        if lite_model:
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()      
        else:
            # load empty model
            self.model = get_empty_model()
            # load weights
            self.model.load_weights(weights_path)
            
            # pre-warm
#            self.predict('data/Food_Classification/chole_bhature/002.jpg')
   

        
    def predict(self, image_path, lite_model=True):
        # pre-process the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = np.reshape(img, [1,IMG_SIZE[0], IMG_SIZE[1], 3])
        img = np.array(img, dtype=np.float32)
        img = self.preprocess_input(img)
        
        if self.lite_model:
            # Prepare input data
            input_data = img  # Your input data, formatted according to input_details
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            # Run inference
            self.interpreter.invoke()

            # Get the output
            preds_out = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:        
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