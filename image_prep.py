import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
import cv2
from scipy.stats import zscore

# Constants
IMG_SIZE = (224, 224)  
RANDOM_STATE = 42

def save_mapping(dict, path="data/pre_processed/mapping.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(dict, f)
        
def get_mapping(path="data/pre_processed/mapping.pkl"):
    with open('saved_dictionary.pkl', 'rb') as f:
        dict = pickle.load(f)
    return dict
    

def remove_outliers(indf):
    """
    Borrowed some outlier detection from: https://www.kaggle.com/code/varsha300/transferlearning
    """
    # Define a threshold for what we consider to be an outlier.
    # We select 3 as that should capture 99.7% of the variance.
    def z_filter(df, col, threshold=3):
        return df[~((df[col] > threshold) | (df[col] < -threshold))]
    df = indf.copy()
    
    # read images
    df['image_gray'] = df.apply(lambda row: cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE), axis=1)
    # remove rows with any non-images
    df = df.dropna(subset=['image_gray'])

    # Filter image sizes based on zscore outliers
    df['image_size'] = df.apply(lambda row: row['image_gray'].shape[0] * row['image_gray'].shape[1] , axis=1)
    df['z_scores_size'] = zscore(df['image_size'])
    df = z_filter(df, 'z_scores_size').copy()

    # Calculate z-scores for the image quality using Laplacian variance
    df['laplacian'] = df.apply(lambda row: cv2.Laplacian(row['image_gray'], cv2.CV_64F).var() , axis=1)
    df['z_scores_laplacian'] = zscore(df['laplacian'])
    df = z_filter(df, 'z_scores_laplacian')    
    return df[['label','path']]


def get_img_df(image_path, pickle_path):
    """
    List the images under each folder and append the path onto each image name.
    Then remove outliers.
    
    Note: removing outliers is an expensive operation.  Cache results for a given path and use those if available.
    """    
    # ensure path exists
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)    
        
    # check for cache
    cache_path = os.path.join(pickle_path, 'cv_fc_no_outliers.pkl.xz') 
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path) 
    
    # each folder is also the label name
    labels = os.listdir(image_path)

    # associate the image path with the label
    image_dct = {}
    for label in labels:
        label_path = os.path.join(image_path, label)
        # read all the files in the path and use the label as the key
        image_dct[label] = [os.path.join(label_path, img) for img in os.listdir(label_path)]
        
    # create df and melt to have the column names (labels) in same row as image path
    # apply a neat trick for dictionaries of unequal lengths
    # https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-have-different-lengths
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in image_dct.items() ])).melt()\
        .dropna().rename(columns={'variable':'label', 'value':'path'})
    df = remove_outliers(df)
    
    # cache results of outlier removal
    df.to_pickle(cache_path)
    
    return df


def process_images_for_CV(list_of_paths, preprocess_input):
    """
    Take a list of paths to images and return a pre-processed set of images.
    Steps adapted from: https://www.kaggle.com/code/varsha300/transferlearning
    Note: I compared the times for iterating over a list and using series.apply(), since this is most likely a pandas series
          but I found almost no difference in performance.  Since iterating over a list will also allow me to take in a single
          image and easily convert it to a list, I'm going the iteration route.
    
    Inputs:
    * list_of_paths: paths to images.  If just a single image is provided, it will be converted to a list.
    * preprocess_input: specific function that is dependent on which keras model was used  
    """
    # ensure we have a list
    list_of_paths = [list_of_paths] if isinstance(list_of_paths, str) else list_of_paths    
    
    images = []
    for path in list_of_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)    
        
    # use model specific pre-processing
    return preprocess_input(images)



def split(X,y,labels, test_size):
    # stratify on each label to ensure we don't introduce a class imbalance.
    return train_test_split(X,y,labels, test_size=test_size, random_state=RANDOM_STATE, stratify=labels)


def train_val_test_split(X,y,labels, test_size=0.1, val_size=0.1):    
    # test validate size will be the combination of those two values
    test_size_c = test_size+val_size    
    # split
    X_train, X_test_validate, y_train, y_test_validate, _, labels_test_validate = split(X,y,labels, test_size_c)
    
    # need to calcuate proportion of test_validate that is attributed to the test set
    test_size_c = test_size / (test_size+val_size)
    # split
    X_test, X_validate, y_test, y_validate, _, _ = split(X_test_validate, y_test_validate, labels_test_validate, test_size_c)
    return X_train, y_train, X_test, y_test, X_validate, y_validate

    
def get_split_data(preprocess_input, model_name, image_path='data/Food_Classification/', pickle_path="data/pre_processed/"):   
    """
    This method is the main entry point for getting test, train, and validation data.
    Cache prep in a pickle file to speed up multiple runs of this for testing    

    Inputs:
        preprocess_input: model specific funciton for processing data
        model_name: needed for pickling the correct version of the model
        image_path: where to look for images
        pickle_path: where to save cached results
    
    """
    # ensure path exists
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)    
    
    # get dataframe of labels, paths, and processed images
    pickle_file_name = f'{model_name}_processed.pkl.xz'
    cache_path = os.path.join(pickle_path, pickle_file_name)
    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path) 
    else:
        # get images as paths
        df = get_img_df(image_path, pickle_path)
        df['images'] = process_images_for_CV(df['path'], preprocess_input)
        df.to_pickle(cache_path)

    # need to convert output to one-hot encocding for ML
    # keras on-hot encoder (to_categorical) requires that labels be integers first.
    # As such, we need to first use a LabelEncoder        
    X = df['images']
    labels = df['label']
    encoder = LabelEncoder().fit(labels)
    y = to_categorical(encoder.transform(labels))
    
    # ensure we also have a reverse mapping
    unique_labels = labels.unique()
    reverse_mapping = {key:val for key,val in zip(encoder.transform(unique_labels), unique_labels)}
    # save for later
    save_mapping(reverse_mapping)

    # stratified split on label
    X_train, y_train, X_test, y_test, X_validate, y_validate = train_val_test_split(X, y, labels, test_size=0.1, val_size=0.1)
    return X_train, y_train, X_test, y_test, X_validate, y_validate
