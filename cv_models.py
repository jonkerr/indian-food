"""
Computer Vision (CV) Models

Generalize architecture and simplify 
"""
import gc

from image_prep import get_split_data
from image_gen import get_gen_from_df

import tensorflow as tf
import keras
from keras.applications import Xception, VGG16, ResNet152V2, InceptionResNetV2, NASNetLarge, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Input
# from keras.models import Model
from keras.optimizers import Adam

# Set the mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

"""
Test harness code to try a variety of model options
"""
# Constants
IMG_SIZE = (224, 224)  # VGG16 default image size

# Base tensor shape on IMG_SIZE
input_tensor_shape = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))


class TastyModel(keras.Model):
    # More info can be found here: https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
    def __init__(self, base_model, num_classes=80, hidden_size=4096, dropout=0.2):
        # tips for dropout: https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
        # Generally, use a small dropout value of 20%-50% of neurons, with 20% providing a good starting point.
        # A probability too low has minimal effect, and a value too high results in under-learning by the network.
        super().__init__()
        # don't train the param on the base model
        self.base_model = base_model

        # don't train base model        
        self.base_model.trainable = False
        
        # add the layers that we'll use for fine tuning
        self.trainable_model = keras.Sequential(
            [
                Flatten(),
                # hidden layer 1
                Dense(hidden_size, activation="relu"), Dropout(dropout), BatchNormalization(),
                # hidden layer 2
                Dense(hidden_size*.5, activation="relu"), Dropout(dropout), #BatchNormalization(),
                # output
                Dense(num_classes)
            ]
        )

    def call(self, inputs):
        x = self.base_model(inputs, training=False)
        return self.trainable_model(x)

def get_tasty_model(base_model, num_classes=80, hidden_size=1024, dropout=0.2):
    """
    Architecture largely from: https://keras.io/guides/transfer_learning/
    """
    # don't train base model        
    for layer in base_model.layers:
        layer.trainable = False

    # Create new model on top
    inputs = input_tensor_shape
    
    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(inputs)
    
    """
    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs)
    """
    
    from tensorflow.keras.models import Model

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    #  Architecture from https://www.kaggle.com/code/varsha300/transferlearning
    #    x = base_model (x, training=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(hidden_size, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = Dense(hidden_size, activation='relu')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # This is the model we will train
    
    return Model(inputs=base_model.input, outputs=predictions)



# Select some candidate models with a high top-5 accuracy
"""
base_models = {
    'Xception': Xception(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    'VGG16': VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    'ResNet152V2': ResNet152V2(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    'InceptionResNetV2': InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    'NASNetLarge': NASNetLarge(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    'EfficientNetB4': EfficientNetB4(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    'EfficientNetB5': EfficientNetB5(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    'EfficientNetB6': EfficientNetB6(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
    'EfficientNetB7': EfficientNetB7(weights='imagenet', include_top=False, input_tensor=input_tensor_shape),
}
"""
base_model_names = [
    'Xception',
    'VGG16',
    'ResNet152V2',
    'InceptionResNetV2',
    'NASNetLarge',
    'EfficientNetB4',
    'EfficientNetB5',
    'EfficientNetB6',
    'EfficientNetB7'
]

def get_base_model(base_name):
    if base_name == 'Xception': 
        return Xception(weights='imagenet', include_top=False, input_tensor=input_tensor_shape)    
    if base_name == 'VGG16': 
        return VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor_shape)
    if base_name == 'ResNet152V2': 
        return  ResNet152V2(weights='imagenet', include_top=False, input_tensor=input_tensor_shape)
    if base_name == 'InceptionResNetV2': 
        return  InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor_shape)
    if base_name == 'NASNetLarge': 
        return  NASNetLarge(weights='imagenet', include_top=False, input_tensor=input_tensor_shape)
    if base_name == 'EfficientNetB4': 
        return  EfficientNetB4(weights='imagenet', include_top=False, input_tensor=input_tensor_shape)
    if base_name == 'EfficientNetB5': 
        return  EfficientNetB5(weights='imagenet', include_top=False, input_tensor=input_tensor_shape)
    if base_name == 'EfficientNetB6': 
        return  EfficientNetB6(weights='imagenet', include_top=False, input_tensor=input_tensor_shape)
    if base_name == 'EfficientNetB7': 
        return  EfficientNetB7(weights='imagenet', include_top=False, input_tensor=input_tensor_shape)



dropout_vals = [0.2, 0.3, 0.4, 0.5]

"""
Helper methods
"""
def get_training_data(df_train, df_validate, batch_size):
    train_gen = get_gen_from_df(df_train, batch_size=batch_size)
    validate_gen = get_gen_from_df(
        df_validate, train=False, batch_size=batch_size)
    # not sure what to do with test data yet so just handle train/test for now
    return train_gen, validate_gen


def train_transfer_model(base_model, df_train, df_validate, epochs=10, num_classes=20, hidden_size=4096, dropout=0.2, batch_size=32, bulk_train=False, lr=0.001):
    # create generators from DFs to ensure same starting place
    train_gen, validate_gen = get_training_data(
        df_train, df_validate, batch_size)
    # Set num steps based on: https://stackoverflow.com/questions/59864408/tensorflowyour-input-ran-out-of-data
    steps_per_epoch = len(df_train)//batch_size
    validation_steps = len(df_validate)//batch_size
    # create model
    model = TastyModel(base_model, dropout=dropout,
                       num_classes=num_classes, hidden_size=hidden_size)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # Train model
    history = model.fit(train_gen, validation_data=validate_gen, epochs=epochs,
                        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    val_loss, val_accuracy = model.evaluate(validate_gen)
    
    # free up memory
    del(train_gen, validate_gen)
    if bulk_train:
        model = None
    gc.collect()
    
    print(f'Validation accuracy: {val_accuracy * 100:.2f}%')
    return history, model, val_loss, val_accuracy


def train_transfer_model_2(base_model, df_train, df_validate, epochs=10, num_classes=20, hidden_size=1024, dropout=0.2, batch_size=32, bulk_train=False, lr=1e-4, num_reductions=2):
    from keras import backend as K
    
    # create generators from DFs to ensure same starting place
    train_gen, validate_gen = get_training_data(
        df_train, df_validate, batch_size)
    # Set num steps based on: https://stackoverflow.com/questions/59864408/tensorflowyour-input-ran-out-of-data
    steps_per_epoch = len(df_train)//batch_size
    validation_steps = len(df_validate)//batch_size
    # create model
    model = get_tasty_model(base_model, num_classes=num_classes, hidden_size=hidden_size, dropout=dropout)    
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    """    class EpocCounterCallback(keras.callbacks.Callback):
        def __init__(self):
            self.current_counter = 0
            self.previous_epochs = 0
        
        def on_epoch_end(self, epoch, logs=None):
            self.current_counter = epoch
            
        def next_run(self):
            self.previous_epochs = self.current_counter
            self.current_counter = 0
        
    
    counter = EpocCounterCallback()"""
    
    # early stopping
#    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-6)
        
    # Train model
    history = model.fit(train_gen, validation_data=validate_gen, epochs=epochs,
                        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[reduce_lr])
    

    """
    num_epocs = len(list(history.history.values())[0])
    print(f'LR {lr} converged after {num_epocs} epochs')

    # iterate over smaller learning rates
    # https://stackoverflow.com/questions/59737875/keras-change-learning-rate
    prev_epocs = num_epocs
    iter_lr = lr
    for _ in range(num_reductions):
        counter.next_run()
        iter_lr = iter_lr / 10    
        # Change learning rate to 0.001 and train for x more epochs
        K.set_value(model.optimizer.learning_rate, iter_lr)
        
        history = model.fit(train_gen, validation_data=validate_gen, epochs=epochs, initial_epoch=counter.previous_epochs,
                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[callback, counter])
        histories.append(history)

        num_epocs = len(list(history.history.values())[0])
        print(f'LR {lr} converged after {num_epocs} additional epochs')
        prev_epocs += num_epocs
    """
    
    val_loss, val_accuracy = model.evaluate(validate_gen)
    
    # free up memory
    del(train_gen, validate_gen)
    if bulk_train:
        model = None
    gc.collect()
    
    print(f'Validation accuracy: {val_accuracy * 100:.2f}%')
    return history, model, val_loss, val_accuracy



def find_best_model(epochs=10, batch_size=32):
    # Enable Dynamic Memory Allocation to deal with OOM issues training multiple models
    # https://www.linkedin.com/pulse/solving-out-memory-oom-errors-keras-tensorflow-running-wayne-cheng
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)    
    
    best_acc = 0
    best_model_name = None
    best_dropout = 0
    best_hidden = 0
    all_results = {}
    
    hidden_sizes= [500, 800, 1000] 
    
    # split once to ensure all models are using the same splits (and ensure nobody is using data in df_test for training, to prevent leakage)
    df_train, df_validate, df_test = get_split_data()
    del(df_test)
    for hidden in hidden_sizes:
        for dropout in dropout_vals:
            for model_name in base_model_names:
                base_model = get_base_model(model_name)
                vals = f'Model_{model_name}__Dropout_{dropout}__HiddenSize_{hidden}'
                print(f'**********\nTraining: {vals}**********\n')
                history, model, val_loss, val_accuracy = train_transfer_model(
                    base_model, df_train, df_validate, epochs=epochs, dropout=dropout, hidden_size=hidden, 
                    batch_size=batch_size, bulk_train=True)

                # free up model memory
                del(model, base_model)
                gc.collect()
                
                all_results[vals] = (history, val_loss, val_accuracy)
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    best_model_name = model_name
                    best_dropout = dropout     
                    best_hidden = hidden               
                    print(f"Best Accuracy: {best_acc}")
                    print(f"Best validation set: {vals}")
                yield best_acc, best_model_name, best_dropout, best_hidden, all_results



def predict(path_to_image, top=3):
    import keras
    import cv2
    import numpy as np
    
    import os
    labels = os.listdir('data/Food_Classification/')
    labels = [l for l in labels if not l.endswith('xz')]

    model_path = 'models/cv_model.keras'
    model = keras.models.load_model(model_path)

    # reshape image
    image = cv2.imread(path_to_image)
    image = cv2.resize(image,(224,224))
    image = np.reshape(image,[1,224,224,3])
    
    preds_out = model.predict(image) 
    softmax = keras.layers.Softmax()
    sm_preds = softmax(preds_out)[0]

    idx = np.argsort(sm_preds)[::-1]

    return [labels[i] for i in idx[:top]]   

