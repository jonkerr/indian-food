# Tasty_AI
### Description
1. Use multiclass image classification to identify Indian food dishes
2. Based on predicted label and other inputs, recommend upto top 5 recipes and similar dishes
3. Using available feedback refine recommendations via reinforcement learning

### Disclaimer
All recommendations are for demonstration purposes only and shouldn't be dependeded on for allergies or other dietary constraints.  Please use your best judegment with your own diet.


### Conda Setup
We **strongly** recommend using conda over venv because conda allows us to specify the correct version of Python.  Assuming you're going to take our good guidance, the conda steps are as follows:

Install environment from conda file
```
conda env create -f environment.yml
```

Then activate
```
conda activate Tasty_AI
```
Easy peasy.

### venv Setup
Conda isn't for everyone.. we get it.  However, this path is a bit harder.  Specifically, this won't work for all versions of Python and has only been tested successfully on **Python 3.9.15**.

To confirm you're using the correct version of Python
```
python --version
```
Anything other than **3.9.15** and we can't guarantee the requirements.txt is going to work.  Your best bet is to ensure you have that version installed on your venv.

Once you have the correct verison of python, it's actually pretty easy.

```
pip install -r requirements.txt
pip install -r requirements2.txt
```
Why two files?  Well, that's another issue that would have been addressed with the conda route.  In order to use a GPU on a Windows machine, we need to use tensorflow==2.10.0.  Unfortunately, this version isn't compatible with Streamlit 1.40, which we needed to address some Streamlit bugs.  Sequentially running these files is a little hack that gets around this issue.

Also, pip *will* complain about incompatible versions of protobuf.  This is expected and accepted.  You can ignore this warning.


#### TL;DR
This would have been easier with conda.  Just sayin.


## Application

This is s streamlit application.  To run, execute the following command:
```
streamlit run tastyai_ui.py
```
It should be self explanatory from there.  Click around and have some fun.  


## Data Sources

**Data Access**
All external data sources are open datasets, found on Kaggle.

**Food_Classification**
* https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification?resource=download-directory

**Recipe data**
 * https://www.kaggle.com/code/cardata/indian-food-cuisine-data-analysis/input
 * https://www.kaggle.com/code/amankumar2002/image-to-recipe/input


### Modules 
Here are the modules used to run the application 

| Application Area | File | Description |
| --- | --- |--- |
|Recommendation Engine|recommendation_models.py|Modular functions used in recommendation system|
|Recommendation Engine|save_load_recommendation_models_vectorizers.py|save and load models and vectorizers used in recommendation system|
|Image Classification|cv_model.py|Define the computer vision model|
|Image Classification|cv_predict.py|Wrapper class to simplify classification predictions|
|Image Classification| image_prep.py | Tools for retriving image files and converting to pandas dataframe |
|Reinforcement Learning|feedback_reinf_learn_recommendation_model.py|Class to adjust recipe recommendations ranking using relevant feedback and save feedback in user_feedback.json file|
|Web Application|Dockerfile|Define what goes into the docker container|
|Web Application|.dockerignore|Define what *doesn't* goes into the docker container|
|Web Application|tastyai_ui.py|Streamlit based file to create TastyAI webpage. This primarily defines the view and processes user input by calling relevant functions|
|Web Application|ui_functions.py|This implements various utility functions to enable the user interaction logic for the TastyAI system in the Streamlit UI|
|Repository Management|.gitignore|Specify files to exclude from adding to git|
|Environment Management|envrionment.yml|Conda environment file to ensure all packages are available|



### Jupyter Notebooks (Examples)
In addition to the modules, it can be helpful to have a variety of examples to have a detailed view as to why certain technical decisions were made.

| Application Area | File | Description |
| --- | --- |--- |
|Image Classification| Example_CV1_Train._Classifier.ipynb | Train and evaluate image classifier model |
|Recommendation Engine|example_compare_recommendation_models.ipynb|test an example to know how the recommendation system works|
|Recommendation Engine|merge_clean_recipe_data.ipynb|Clean the merged recipe dataset|
|Recommendation Engine|preprocess_recipe_data.ipynb|preprocess the recipe data|
|Recommendation Engine|topic_modeling_NMF.ipynb|Train NMF model on TFIDF and count vectorizers using various combination of data|
|Recommendation Engine|topic_modeling_SVD.ipynb|Train SVD model on TFIDF and count vectorizers using various combination of data|


### Working Data
There are three categories of persisted data: 
* Source Data - this is data that is downloaded from the data sources listed above
* Model Data - this is data has been constructed and serialized for future use
* Feedback Data - user feedback data is collected and utilized by Feedback Reinforcement Learning re-ranking

#### Source Data
| Usage | Path |
| --- | --- |
|Image classifier training data| data/Food_Classification|
|First recipe dataset | data/Indian_Food_Cuisine.csv|
|Second recipe dataset | data/Indian_Food_Recipe.csv|
|Merged and cleaned recipe dataset| data/nohindi_noduplicate_recipes.pkl|
|Processed recipe dataset | data/processed_recipes.pkl|


#### Model Data
Note: This data may not actually exist as it is too large to check into GitHub.  Instead, it will be generated by running the code in the notebooks.  The status column will let the reader know if it exists in GitHub or needs to be generated by running the code.

| Usage | Status | Path |
| --- | --- | --- |
|Image classifier weights| Generated | models/weights/efficientnet_v2_20_84.64.hdf5|
|Image classifier - TF Lite model| Generated | models/lite/efficientnet_v2_20_84.64.tflite|
|Recomendation Count Vectorizer| Checked in | models/count_vectorizer.pkl|
|Recomendation NMF Count Model|Checked in | models/nmf_count_model.pkl|
|Recomendation NMF TF-IDF Model|Checked in | models/nmf_count_model.pkl|
|Recomendation Vectorizer Count Model|Checked in | models/tfidf_vectorizer.pkl|
|Recomendation Vectorizer TF-IDF Model|Checked in | models/tfidf_vectorizer.pkl|


#### Feedback Data
This is cleaned user feedback data generated using our integration testing and demo to demonstrate how feedback is saved. This will be helpful for new users to test if the reranking is working accurately when the feedback exists for that recipe. This will be updated with new feedback anytime a user provids a feedback on the screen. 

| Usage | Status | Path |
| --- | --- | --- |
|User Feedback Collection| Checked in and also Generated | models/user_feedback.json|