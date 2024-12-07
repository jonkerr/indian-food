# Force CPU only for predictions
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow 
import cv2
import numpy as np
from image_prep import get_label_mapping, IMG_SIZE
from cv_model import get_empty_model

class TastyFoodPredictor():
    def __init__(self, 
                 preprocess_input_fn=None, 
                 tflite_path='models/lite/efficientnet_v2_20_84.64.tflite', 
                 weights_path = 'models/weights/efficientnet_v2_20_84.64.hdf5', 
                 use_keras=False, use_tflite=True):
           
        # use a mapping dictionary to find the values associated with a prediction     
        self.mapping_dct = get_label_mapping()

        # load specific preprocess_input or use the default if not specified
        self.preprocess_input = preprocess_input_fn
        if preprocess_input_fn is None:
            self.preprocess_input = tensorflow.keras.applications.efficientnet_v2.preprocess_input
            
        # we'll default to using TF Lite but still have the option to turn off
        self.use_tflite = use_tflite
        if use_tflite:
            # Load the TFLite model
            self.interpreter = tensorflow.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()      
            
        # only load keras model if asked as this is very expensive!!
        if use_keras:
            # load empty model
            self.model = get_empty_model()
            # load weights
            self.model.load_weights(weights_path)
            # pre-warm
            self.predict('data/Food_Classification/chole_bhature/002.jpg', use_tflite=False)
   

        
    def predict(self, image_path, use_tflite=True):
        # pre-process the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = np.reshape(img, [1,IMG_SIZE[0], IMG_SIZE[1], 3])
        img = np.array(img, dtype=np.float32)
        img = self.preprocess_input(img)
        
        # predict
        preds_out = self.predict_batch(img, use_tflite)
        
            # map to label
        idx = np.argmax(preds_out[0])
        return self.mapping_dct[idx]
    
    
    def predict_batch(self, input_data, use_tflite=True):
        if use_tflite:            
            # got weird shape errors when predicting a batch of records.  
            # Found a workaround here:
            # https://heartbeat.comet.ml/running-tensorflow-lite-image-classification-models-in-python-92ef44b4cd47 
            self.interpreter.resize_tensor_input(self.input_details[0]['index'],[len(input_data), IMG_SIZE[0], IMG_SIZE[1], 3])
            self.interpreter.allocate_tensors()
            
            # Prepare input data
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            # Run inference
            self.interpreter.invoke()

            # Get the output
            preds_out = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:        
            # predict
            preds_out = self.model.predict(input_data) 
            
        return preds_out

    
    
def health_check():       
    print('starting health check')
    test_image = 'data/Food_Classification/chole_bhature/002.jpg'
    # confirm key files exist
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