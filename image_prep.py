import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def clean_df(df):
    """
    Insert cleaning code here
    """
    return df

def get_img_df(path):
    """
    List the images under each folder and append the path onto each image name
    """
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
        
    return clean_df(df)


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
