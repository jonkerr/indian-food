import pandas as pd
import numpy as np
import os
import pickle
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from recommendation_models import compute_coherence_scores


#Save models trained in the recommendation system
def save_model(model, file_name, path='models'):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(model, f)

#Save vectorizers used in the recommendation system
def save_vectorizer(vectorizer, file_name, path='models'):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(vectorizer, f)   

def get_recommendations_save_pickles(filtered_df, vectorizer, model, file_name, path, num_recommendations=5):
    # Compute the optimal number of topics
    best_num_topics = compute_coherence_scores(filtered_df, vectorizer, model)

    # Transform the text data into a TF-IDF matrix
    vectorizer_matrix = vectorizer.fit_transform(filtered_df['combined_name_ingredients'])

    # Ensure all values in the TF-IDF matrix are non-negative and finite
    vectorizer_matrix = np.nan_to_num(vectorizer_matrix.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    # Train the NMF model with the optimal number of topics
    called_model = model(n_components=best_num_topics, random_state=42)
    called_model.fit(vectorizer_matrix)
    topic_matrix = called_model.transform(vectorizer_matrix)  # Topic distribution for each recipe

#   Save the vectorizer and model
    #save_vectorizer(vectorizer, file_name, path)
    #save_model(called_model, file_name, path)
    
    # Copy the filtered DataFrame to avoid overwriting the original
    filtered_df = filtered_df.copy()

    # Add topic distributions to the filtered DataFrame
    for i in range(best_num_topics):
        filtered_df[f'topic_{i}'] = topic_matrix[:, i]

    # Compute cosine similarities for all dishes
    cosine_similarities = cosine_similarity(topic_matrix)

    if len(filtered_df) > num_recommendations:
        # More recipes than the number of recommendations
        highest_avg_score = 0.0
        best_set_indices = []

        # Loop through each dish and calculate the average similarity of its top similar dishes
        for dish_index in range(len(filtered_df)):
            # Get indices of the most similar recipes for the current dish
            similar_indices = cosine_similarities[dish_index].argsort()[-num_recommendations-1:-1][::-1]

            # Compute the average similarity score for the top similar dishes
            avg_similarity = cosine_similarities[dish_index][similar_indices].mean()

            # Update the best set of dishes if the current average similarity is higher
            if avg_similarity > highest_avg_score:
                highest_avg_score = avg_similarity
                best_set_indices = similar_indices
                best_similarity_scores = cosine_similarities[dish_index][similar_indices]

        # Create a DataFrame with the top recommended dishes
        recommended_recipes = filtered_df.iloc[best_set_indices].copy()
        recommended_recipes['similarity_score'] = [f"{(score * 100):.2f}%" for score in best_similarity_scores]

        # Print the highest average similarity score
        print(f"\nHighest average similarity score among all dishes: {highest_avg_score:.4f}\n")

        return None

    else:
        # Less than or equal to the number of recommendations
        base_recipe_index = 0  # Use the first recipe as the base
        similarity_scores = cosine_similarities[base_recipe_index]
        avg_similarity = similarity_scores.mean()

        # Add similarity scores to the DataFrame
        filtered_df['similarity_score'] = [f"{(score * 100):.2f}%" for score in similarity_scores]

        # Print the average similarity score
        print(f"\nAverage similarity score across all dishes: {avg_similarity:.4f}\n")

        return None

#Load models trained in the recommendation system
def get_model(file_name, path='models'):
     
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file '{file_name}' does not exist in the path '{path}'.")
    
    with open(full_path, 'rb') as f:
        return pickle.load(f)
    
#Load vectorizers used in the recommendation system
def get_vectorizer(file_name, path='models'):
     
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file '{file_name}' does not exist in the path '{path}'.")
    
    with open(full_path, 'rb') as f:
        return pickle.load(f)