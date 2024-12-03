# Dockerfile is a blueprint for building images
# An image is template for running containers
# Container is the running process
FROM python:3.9.7-slim

# Packages
RUN pip install spacy==3.5.3 matplotlib==3.9.2 numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 seaborn==0.13.2 scikit-learn==1.5.1 nltk==3.9.1 gensim==4.3.3 tensorflow==2.10.0 opencv-python==4.10.0.84 streamlit==1.9.0 altair==4.2.2

# Test picture
RUN mkdir -p ./data/Food_Classification/chole_bhature/
ADD data/Food_Classification/chole_bhature/002.jpg ./data/Food_Classification/chole_bhature/

# Add CV models
RUN mkdir  -p ./models/weights
ADD ./models/efficientnet_v2_20_84.64.keras ./models/
ADD ./models/weights/efficientnet_v2_20_84.64.hdf5 ./models/weights/

# Add source data
RUN mkdir -p ./data/pre_processed/
ADD data/nohindi_recipes.pkl data/
ADD data/processed_recipes.pkl data/
ADD data/pre_processed/mapping.pkl data/pre_processed/

# Add CV project files
#ADD image_prep.py .
#ADD cv_predict.py .
#ADD cv_model.py .

#Add recommender models .
#ADD recommendation_models.py .

# Copy recommendation models
COPY models/*.pkl models/

# We don't have the version for headless so install last as
RUN pip install opencv-python-headless

# Add UI (most likely to change)
ADD *.py .

#EXPOSE 8501

CMD ["streamlit", "run", "tastyai_ui.py", "--server.port", "80"]