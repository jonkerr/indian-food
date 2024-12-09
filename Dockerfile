# Dockerfile is a blueprint for building images
# An image is template for running containers
# Container is the running process
FROM python:3.9.15-slim

# Packages
RUN pip install jupyter==1.0.0 altair==5.0.1 matplotlib==3.9.2 numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 seaborn==0.13.2 spacy==3.5.3 scikit-learn==1.5.1 nltk==3.9.1 gensim==4.3.3 wordcloud==1.9.3 plotly==5.24.1 tensorflow==2.10.0 opencv-python==4.10.0.84
# This only installs if it happens after the others
RUN pip install protobuf==3.20.* streamlit==1.40.1

# Test picture
RUN mkdir -p ./data/Food_Classification/chole_bhature/
ADD ./data/Food_Classification/chole_bhature/002.jpg ./data/Food_Classification/chole_bhature/

# Add CV models
RUN mkdir  -p ./models/weights
RUN mkdir  -p ./models/lite
ADD ./models/weights/efficientnet_v2_20_84.64.hdf5 ./models/weights/
ADD ./models/lite/efficientnet_v2_20_84.64.tflite ./models/lite/

# Add source data
RUN mkdir -p ./data/pre_processed/
ADD ./data/nohindi_noduplicate_recipes.pkl ./data/
ADD ./data/processed_recipes.pkl ./data/
ADD ./data/pre_processed/mapping.pkl ./data/pre_processed/

# Copy recommendation models
COPY ./models/*.pkl ./models/

# We don't have the version for headless so install last as
RUN pip install opencv-python-headless

# Add UI (most likely to change)
ADD *.py .

CMD ["streamlit", "run", "tastyai_ui.py", "--server.port", "80"]