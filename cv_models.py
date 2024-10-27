"""
Computer Vision (CV) Models

Generalize architecture and simplify 
"""

from image_prep import get_split_data
from image_gen import get_gen_from_df

# import tensorflow as tf
import keras
from keras.applications import Xception, VGG16, ResNet152V2, InceptionResNetV2, NASNetLarge, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Input
# from keras.models import Model
from keras.optimizers import Adam


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
                Dense(hidden_size, activation="relu"), Dropout(dropout), #BatchNormalization(),
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


def train_transfer_model(base_model, df_train, df_validate, epochs=10, num_classes=20, hidden_size=4096, dropout=0.2, batch_size=32):
    # create generators from DFs to ensure same starting place
    train_gen, validate_gen = get_training_data(
        df_train, df_validate, batch_size)
    # Set num steps based on: https://stackoverflow.com/questions/59864408/tensorflowyour-input-ran-out-of-data
    steps_per_epoch = len(df_train)//batch_size
    validation_steps = len(df_validate)//batch_size
    # create model
    model = TastyModel(base_model, dropout=dropout,
                       num_classes=num_classes, hidden_size=hidden_size)
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # Train model
    history = model.fit(train_gen, validation_data=validate_gen, epochs=epochs,
                        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    val_loss, val_accuracy = model.evaluate(validate_gen)
    print(f'Validation accuracy: {val_accuracy * 100:.2f}%')
    return history, model, val_loss, val_accuracy


def find_best_model(epochs=10):
    best_acc = 0
    best_model_name = None
    best_dropout = 0
    best_hidden = 0
    all_results = {}
    
    hidden_sizes= [20,50,100,200] 
    
    # split once to ensure all models are using the same splits (and ensure nobody is using data in df_test for training, to prevent leakage)
    df_train, df_validate, df_test = get_split_data()
    for model_name, base_model in base_models.items():
        for dropout in dropout_vals:
            for hidden in hidden_sizes:
                vals = f'Model_{model_name}__Dropout_{dropout}__HiddenSize_{hidden}'
                print(f'**********\nTraining: {vals}**********\n')
                history, model, val_loss, val_accuracy = train_transfer_model(
                    base_model, df_train, df_validate, epochs=epochs, dropout=dropout, hidden_size=hidden)
                all_results[vals] = (history, model, val_loss, val_accuracy)
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    best_model_name = model_name
                    best_dropout = dropout     
                    best_hidden = hidden               
                    print(f"Best Accuracy: {best_acc}")
                    print(f"Best validation set: {vals}")
    return best_acc, best_model_name, best_dropout, best_hidden, all_results
