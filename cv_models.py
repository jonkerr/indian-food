"""
Computer Vision (CV) Models

Generalize architecture and simplify 
"""

from image_prep import get_split_data
from image_gen import get_gen_from_df

import tensorflow as tf 
import keras
from keras.applications import Xception, VGG16, ResNet152V2, InceptionResNetV2, NASNetLarge, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Input
#from keras.models import Model
#from keras.optimizers import Adam


class TastyModel(keras.Model):
    # More info can be found here: https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
    
    def __init__(self, base_model, num_classes=80, hidden_size=4096, dropout=0.2):
        # tips for dropout: https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
        # Generally, use a small dropout value of 20%-50% of neurons, with 20% providing a good starting point. 
        # A probability too low has minimal effect, and a value too high results in under-learning by the network.
        
        super().__init__()
        
        # don't train the param on the base model 
        self.base_model = base_model
        base_model.trainable = False
        
        # add the layers that we'll use for fine tuning
        self.trainable_model = keras.Sequential(
            [
                Flatten(),
                # hidden layer 1
                Dense(hidden_size, activation="relu"), Dropout(dropout), BatchNormalization(),
                # hidden layer 2
                Dense(hidden_size, activation="relu"), Dropout(dropout), BatchNormalization(),
                # output
                Dense(num_classes)
            ]
        )
        
        
    def call(self, inputs):
        x = self.base_model(inputs, training=False)
        return self.trainable_model(x)
    
    

"""
Test harness code to try a variety of model options
"""

# Constants
IMG_SIZE = (224, 224)  # VGG16 default image size

# Base tensor shape on IMG_SIZE
input_tensor_shape = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Select some candidate models with a high top-5 accuracy
base_models = [
    Xception(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    ResNet152V2(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    NASNetLarge(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    EfficientNetB4(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    EfficientNetB5(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    EfficientNetB6(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    EfficientNetB7(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
]

dropout_vals = [0.2, 0.3, 0.4, 0.5]

"""
Helper methods
"""

def get_training_data():
    train, validate, test = get_split_data()

    train_gen = get_gen_from_df(train)
    validate_gen = get_gen_from_df(validate, train=False)
    
    # not sure what to do with test data yet so just handle train/test for now
    return train_gen, validate_gen



    