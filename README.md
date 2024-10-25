# indian-food
1. Use computer vision to identify Indian food dishes.
2. Based on derived label, recommend recipes and similar dishes

#### Setup
Install environment from conda file
```
conda env create -f environment.yml
```

Then activate
```
conda activate Tasty_AI
```

#### Data Sources
* Indian_Food_Images : https://www.kaggle.com/code/harshghadiya/transfer-learning-models/input
* Food_Classification: https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification?resource=download-directory


#### Overview of Modules 
* image_prep.py - tools for retriving image files and converting to pandas dataframe
* image_gen.py - tools for converting dataframes to keras image generators

#### Jupyter Notebooks
* simple_example.ipynb - very simple example of end to end image classification run