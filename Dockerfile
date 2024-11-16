# Dockerfile is a blueprint for building images
# An image is template for running containers
# Container is the running process
FROM python:3.9.7

# Can pip install on one line but this was useful for debugging
RUN pip install altair==4.2.2 
RUN pip install matplotlib==3.9.2 
RUN pip install numpy==1.26.4 
RUN pip install pandas==2.2.2 
RUN pip install scipy==1.13.1 
RUN pip install seaborn==0.13.2 
RUN pip install spacy==3.5.3 
RUN pip install scikit-learn==1.5.1 
RUN pip install nltk==3.9.1 gensim==4.3.3 
RUN pip install tensorflow==2.10.0 
RUN pip install opencv-python==4.10.0.84
RUN pip install streamlit==1.12.0

# Add source data
RUN mkdir -p ./data/pre_processed/
ADD data/nohindi_recipes.pkl data/
ADD data/processed_recipes.pkl data/
ADD data/pre_processed/mapping.pkl data/pre_processed/

# Test image
RUN mkdir -p ./data/Food_Classification/chole_bhature/
ADD data/Food_Classification/chole_bhature/002.jpg ./data/Food_Classification/chole_bhature/

# Add CV models
RUN mkdir ./models/
ADD models/efficientnet_v2_20_84.64.keras ./models/

# Add CV project files
ADD image_prep.py .
ADD cv_predict.py .

# We don't have the version for headless so install last as
RUN pip install opencv-python-headless

# Add UI (most likely to change)
ADD tastyai_ui.py .

CMD ["python", "cv_predict.py"]
#CMD ["streamlit", "run", "tastyai_ui.py"]