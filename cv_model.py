
from image_prep import get_split_data, get_label_mapping, IMG_SIZE
from image_gen import train_image_generator

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



"""
Got a fair amount of inspiration for the base approach from:  https://www.kaggle.com/code/varsha300/transferlearning
but have adapted a number of things to make it easier to alter different aspects 
"""

DEBUG = False
def print_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
    
    
def get_keras_model():
    """
    Put this in a separate method to make it easier to try different keras models
    
    Outputs
        base_model: pre-trained keras model for transfer learning
        model_name: name of model for caching pre-processed
        preprocess_input:  model specific preprocess function
    """
    # Imports are specific to the keras model
    from tensorflow.keras.applications import EfficientNetV2L
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    
#    from tensorflow.keras.applications import InceptionResNetV2
#    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    
    model_name = 'efficientnet_v2'
    base_model = EfficientNetV2L(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
        
    base_model.trainable = False
    return base_model, model_name, preprocess_input



def get_tasty_model(base_model, num_classes):
    """
    Get a transfer learning model 
    
    Inputs:
        base_model: keras learning model
        model_name: name of model for caching pre-processed
        preprocess_input:  model specific preprocess function
    """    
    # Model architecture
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)    
    

def train_model(tasty_model, X_train, y_train, X_val, y_val, batch_size, epochs, verbose):
    """
    Get a transfer learning model 
    
    Inputs:
        tasty_model: our transfer learning model
        model_name: name of model for caching pre-processed
        preprocess_input:  model specific preprocess function
    """
    # Compile model
    tasty_model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    data_gen = train_image_generator()
    spe = len(X_train) // 32    
    print_debug(f'steps per epoch: {spe}')    
    flow = data_gen.flow(X_train, y_train, batch_size=batch_size)
    history = tasty_model.fit(flow, validation_data=(X_val, y_val), epochs=epochs, steps_per_epoch=spe, verbose=verbose)
    
    # Evaluate model
    val_loss, val_accuracy = tasty_model.evaluate(X_val, y_val)
    print(f'Validation accuracy: {val_accuracy * 100:.2f}%')
        
    return tasty_model, history, val_loss, val_accuracy



def training_harness(batch_size=32, epochs=20, verbose=1):
    """
    Need a better name :) 
    """
    # get keras model
    print_debug('Getting keras model')
    base_model, model_name, preprocess_input = get_keras_model()
    
    # get data
    print_debug('Getting data')
    mapping_dct = get_label_mapping()
    X_train, y_train, X_test, y_test, X_val, y_val, label_encoder = get_split_data(preprocess_input, model_name)
    print_debug(f'X_train shape: {X_train.shape}')
    print_debug(f'y_train shape: {y_train.shape}')
    print_debug(f'X_val shape: {X_val.shape}')
    print_debug(f'y_val shape: {y_val.shape}')
    print_debug('***************************')
    
    
    # get tasty model
    print_debug('Getting tasty model')
    num_classes = len(mapping_dct.values())
    tasty_model = get_tasty_model(base_model, num_classes)
    
    # train model
    print_debug(f'Training for {epochs} epochs')    
    trained_model, history, val_loss, val_accuracy = train_model(tasty_model, X_train, y_train, X_val, y_val, batch_size, epochs, verbose)
    
    path = f'models/L3_{model_name}_{epochs}_{val_accuracy * 100:.2f}.keras'
    trained_model.save(path)
    
    return trained_model, history, val_loss, val_accuracy, label_encoder
    
