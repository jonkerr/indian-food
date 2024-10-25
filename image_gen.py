import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_basic_image_generator():
    datagen = ImageDataGenerator(rescale=1./255)    
    return datagen

def get_image_generator():
    # abstract the params used in image generation
    datagen = ImageDataGenerator(
        rescale=1./255,        
        rotation_range=20, 
        zoom_range=0.15, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.15, 
#        brightness_range=0.1,
        horizontal_flip=True, 
        vertical_flip=True,
        fill_mode="nearest",
        channel_shift_range=0.1,
    )    
    return datagen
    

def get_gen_from_df(df, target_size=(224, 224), batch_size=32, train=True):
    # TODO: Should there be a color correction step similar to that found in:
    # https://www.kaggle.com/code/varsha300/transferlearning
    #    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # use pre-defined params
    if train:
        datagen = get_image_generator()
    else:
        datagen = get_basic_image_generator()
        

    # create generator from df
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col='path',
        y_col='label',
        target_size=target_size, # TBD
        batch_size=batch_size, # TBD
        class_mode='categorical'  
    )    

""" This doesn't work.  Leaving in for now in case I want to give another chance but will likely be deleted at some point.
def get_gen_from_prefetch(train_ds):

    # use pre-defined params
    datagen = get_image_generator()

    ## Google AI (Gemini?) answered my question on how to map a PreFetchDataset to an ImageDataGenerator
    # https://www.google.com/search?q=tensorflow+ImageDataGenerator+_PrefetchDataset&sca_esv=339a018dea4bc811&ei=ebIRZ8OfIOO4kPIPztnF0Qw&ved=0ahUKEwiDwLvM3JaJAxVjHEQIHc5sMcoQ4dUDCA8&uact=5&oq=tensorflow+ImageDataGenerator+_PrefetchDataset&gs_lp=Egxnd3Mtd2l6LXNlcnAiLnRlbnNvcmZsb3cgSW1hZ2VEYXRhR2VuZXJhdG9yIF9QcmVmZXRjaERhdGFzZXQyChAAGLADGNYEGEcyChAAGLADGNYEGEcyChAAGLADGNYEGEcyChAAGLADGNYEGEcyChAAGLADGNYEGEcyChAAGLADGNYEGEcyChAAGLADGNYEGEcyChAAGLADGNYEGEdIxRxQ0gNY_BVwAXgAkAEAmAE6oAE6qgEBMbgBA8gBAPgBAvgBAZgCAaACA5gDAIgGAZAGCJIHATGgB6EF&sclient=gws-wiz-serp

    # Apply data augmentation
    dataset = train_ds.map(lambda x, y: (datagen.flow(x, batch_size=32), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Add prefetching
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
"""
