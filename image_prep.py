import os
import pandas as pd
from sklearn.model_selection import train_test_split



RANDOM_STATE = 42

def remove_outliers(indf):
    """
    Borrowed some outlier detection from: https://www.kaggle.com/code/varsha300/transferlearning
    """
    import cv2
    from scipy.stats import zscore
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


def get_img_df(path):
    """
    List the images under each folder and append the path onto each image name.
    Then remove outliers.
    
    Note: removing outliers is an expensive operation.  Cache results for a given path and use those if available.
    """
    
    # check for cache
    cache_path = os.path.join(path, 'image_df.pkl.xz') 
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path) 
    
    # each folder is also the label name
    labels = os.listdir(path)

    # associate the image path with the label
    image_dct = {}
    for label in labels:
        label_path = path + label
        # Yes, the preferred method is os.path.join but it's ugly on Windows machines and 
        # *NIX style paths seem to work just fine in python.
        image_dct[label] = [label_path + '/' + img for img in os.listdir(label_path)]
        
    # create df and melt to have the column names (labels) in same row as image path
    # apply a neat trick for dictionaries of unequal lengths
    # https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-have-different-lengths
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in image_dct.items() ])).melt()\
        .dropna().rename(columns={'variable':'label', 'value':'path'})
    df = remove_outliers(df)
    
    # cache results of outlier removal
    df.to_pickle(cache_path)
    
    return df


def split(df, test_size):
    # stratify on each label to ensure we don't introduce a class imbalance.
    return train_test_split(df, test_size=test_size, random_state=RANDOM_STATE, stratify=df['label'])


def train_val_test_split(df, test_size=0.2, val_size=0.2):    
    # test validate size will be the combination of those two values
    test_size_c = test_size+val_size    
    # split
    train, test_validate = split(df, test_size_c)
    
    # need to calcuate proportion of test_validate that is attributed to the test set
    test_size_c = test_size / (test_size+val_size)
    # split
    test, validate = split(test_validate, test_size_c)
    return train, validate, test


def get_split_data(path="data/Food_Classification/"):
    # get images as paths
    df = get_img_df(path)
    
    # split
    train, validate, test = train_val_test_split(df, test_size=0.2, val_size=0.2)
    return train, validate, test
