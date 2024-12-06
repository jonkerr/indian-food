# Tasty_AI
### Description
1. Use multiclass image classification to identify Indian food dishes
2. Based on predicted label and other inputs, recommend recipes and similar dishes
3. Refine recommendations via reinforcement learning

### Setup
Install environment from conda file
```
conda env create -f environment.yml
```

Then activate
```
conda activate Tasty_AI
```

## Application
### Data Sources
* Food_Classification: https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification?resource=download-directory



### Modules 
Here are the modules used to run the application 

| Application Area | File | Description |
| --- | --- |--- |
|Recommendation Engine|recommendation_models.py|** TBD **|
|Recommendation Engine|save_load_recommendation_models_vectorizers.py|** TBD **|
|Image Classification|cv_model.py|Define the computer vision model|
|Image Classification|cv_predict.py|Wrapper class to simplify classification predictions|
|Image Classification| image_prep.py | Tools for retriving image files and converting to pandas dataframe |
|Image Classification| image_gen.py | Tools for converting dataframes to keras image generators |
|Reinforcement Learning|save_user_feedback.py|** TBD **|
|Reinforcement Learning|feedback_reinf_learn_recommendation_model.py|** TBD **|
|Web Application|Dockerfile|Define what goes into the docker container|
|Web Application|.dockerignore|Define what *doesn't* goes into the docker container|
|Web Application|tastyai_ui.py|Streamlit application file.  This file primarily defines the view|
|Web Application|ui_functions.py|Separation of concerns for streamlit application.  This file contains the logical/controller aspects of the web application.|
|Repository Management|.gitignore|Specify files to exclude from adding to git|
|Environment Management|envrionment.yml|Conda environment file to ensure all packages are available|


### Jupyter Notebooks (Examples)
In addition to the modules, it can be helpful to have a variety of examples to have a detailed view as to why certain technical decisions were made.

| Application Area | File | Description |
| --- | --- |--- |
|Image Classification| Example_CV1_Train._Classifier.ipynb | Train and evaluate image classifier model |
|Recommendation Engine|example_compare_recommendation_models.ipynb|** TBD **|
|Recommendation Engine|prepare_recommendation_data.ipynb|** TBD **|
|Recommendation Engine|topic_modeling_NMF.ipynb|** TBD **|
|Recommendation Engine|topic_modeling_SVD.ipynb|** TBD **|




