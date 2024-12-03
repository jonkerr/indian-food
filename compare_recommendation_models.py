#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


processed_df = pd.read_pickle("data/processed_recipes.pkl")
#processed_df


# In[3]:


tfidf_vectorizer = TfidfVectorizer()
count_vectorizer = CountVectorizer()

# Find dish name in recipes and filter them based on user preferences
from recommendation_models import filter_recipes

# filtered_df = filter_recipes (processed_df, "masala dosa", diet = "Vegetarian")
# filtered_df

# Calculate the best number of topics using coherence score for the filtered dataframe
from recommendation_models import compute_coherence_scores

# Train the recommendation model using topic models and vectorizers
from recommendation_models import get_recommendations
from recommendation_models import get_recommendations_nmf_tfidf
from recommendation_models import get_recommendations_nmf_count
from recommendation_models import get_recommendations_svd_tfidf
from recommendation_models import get_recommendations_svd_count

# # Get recipe recommendation using the best NMF model on TF-IDF vectorizer
# recommended_recipes_nmf_tfidf = get_recommendations_nmf_tfidf(filtered_df) 
# # recommended_recipes_nmf_tfidf

# # Get recipe recommendation using best NMF model on Count vectorizer
# recommended_recipes_nmf_count = get_recommendations_nmf_count(filtered_df) 
# # recommended_recipes_nmf_count

# # Get recipe recommendation using best SVD model on TF-IDF vectorizer
# recommended_recipes_svd_tfidf = get_recommendations_svd_tfidf(filtered_df) 
# # recommended_recipes_svd_tfidf

# # Get recipe recommendation using best SVD model on Count vectorizer
# recommended_recipes_svd_count = get_recommendations_svd_count(filtered_df) 
# # recommended_recipes_svd_count

# Compare recommendation models and choose the one with the highest average of similarity score as the best model 

def compare_recommendation_models(filtered_df):
    # Define a dictionary to store model functions and names
    model_functions = {"NMF_TFIDF": get_recommendations_nmf_tfidf, "NMF_Count": get_recommendations_nmf_count,
        "SVD_TFIDF": get_recommendations_svd_tfidf, "SVD_Count": get_recommendations_svd_count }

    results = []
    all_recommendations = {}

    for model_name, model_func in model_functions.items():
        # Get recommendations using the model
        recommended_recipes = model_func(filtered_df)

         # Store the recommendations for the current model
        all_recommendations[model_name] = recommended_recipes

        # Extract similarity scores and calculate the average
        similarity_scores = recommended_recipes['similarity_score'].str.rstrip('%').astype(float) / 100
        average_score = similarity_scores.mean()

        # Append the results to the list
        results.append({"Model": model_name, "Average Similarity Score": average_score})

    # Make a DataFrame for the results for better visualizacomparison
    similarity_df = pd.DataFrame(results)

    # Find the model with the highest average similarity score
    best_model = similarity_df.loc[similarity_df["Average Similarity Score"].idxmax()]
    best_model_name = best_model['Model']
    best_recommendations = all_recommendations[best_model_name]

    print("Comparison Table:")
    print(similarity_df)

    print(f"\nBest Model: {best_model['Model']} with Average Similarity Score: {best_model['Average Similarity Score']:.4f}")

    return best_recommendations

# best_recommendations = compare_recommendation_models(filtered_df)
# best_recommendations

# Save models and vectoriers used in recommendation system
def save_model(model, file_name, path='models'):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(model, f)

def save_vectorizer(vectorizer, file_name, path='models'):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(vectorizer, f)

def get_recommendations_save_pickles(filtered_df, vectorizer, model, file_name, path, num_recommendations=5): 

    best_num_topics = compute_coherence_scores(filtered_df, vectorizer, model)

    vectorizer_matrix = vectorizer.fit_transform(filtered_df['combined_name_ingredients'])

    # Ensure all values in tfidf_matrix are non-negative and finite
    vectorizer_matrix = np.nan_to_num(vectorizer_matrix.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    # Train the NMF model with the optimal number of topics    
    called_model = model(n_components=best_num_topics, random_state=42)
    called_model.fit(vectorizer_matrix)
    topic_matrix = called_model.transform(vectorizer_matrix)  # Topic distribution for each recipe

#   Save the vectorizer and model
    #save_vectorizer(vectorizer, file_name, path)
    #save_model(called_model, file_name, path)
    
    filtered_df = filtered_df.copy()

    # Add topic distributions to the filtered DataFrame
    for i in range(best_num_topics):
        filtered_df[f'topic_{i}'] = topic_matrix[:, i]

    dish_index = filtered_df.reset_index().index[0]  # Get the first match if there are multiple

    cosine_similarities = cosine_similarity(topic_matrix)
    
    # Get indices of the most similar recipes
    similar_indices = cosine_similarities[dish_index].argsort()[-num_recommendations-1:-1][::-1]

    recommended_recipes = filtered_df.iloc[similar_indices].copy()
    recommended_recipes["similarity_score"] = cosine_similarities[dish_index][similar_indices]

    # Scale and format similarity scores to percentages
    recommended_recipes['similarity_score'] = (recommended_recipes['similarity_score'] * 100).round(2).astype(str) + '%'

    return None

#tfidf_vect = get_recommendations_save_pickles(filtered_df, tfidf_vectorizer, NMF, "tfidf_vectorizer.pkl", path='models')
#count_vect = get_recommendations_save_pickles(filtered_df, count_vectorizer, NMF, "count_vectorizer.pkl", path='models')
#nmf_tfid_model = get_recommendations_save_pickles(filtered_df, tfidf_vectorizer, NMF, "nmf_tfidf_model.pkl", path='models')
#nmf_count_model = get_recommendations_save_pickles(filtered_df, count_vectorizer, NMF, "nmf_count_model.pkl", path='models')
#svd_tfidf_model = get_recommendations_save_pickles(filtered_df, tfidf_vectorizer, TruncatedSVD,"svd_tfidf_model.pkl", path='models')
#svd_count_model = get_recommendations_save_pickles(filtered_df, count_vectorizer,TruncatedSVD, "svd_count_model.pkl", path='models')

# Load models and vectoriers used in recommendation system
def get_model(file_name, path='models'):
     
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file '{file_name}' does not exist in the path '{path}'.")
    
    with open(full_path, 'rb') as f:
        return pickle.load(f)

#nmf_tfidf = get_model("nmf_tfidf_model.pkl", "models")
#nmf_tfidf 

#nmf_count = get_model("nmf_count_model.pkl", "models")
#nmf_count

#svd_tfidf = get_model("svd_tfidf_model.pkl", "models")
#svd_tfidf

#svd_count = get_model("svd_count_model.pkl", "models")
#svd_count

def get_vectorizer(file_name, path='models'):
     
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file '{file_name}' does not exist in the path '{path}'.")
    
    with open(full_path, 'rb') as f:
        return pickle.load(f)

#tfidf_vec = get_vectorizer("tfidf_vectorizer.pkl", "models")
#tfidf_vec

#count_vec = get_vectorizer("count_vectorizer.pkl", "models")
#count_vec

# Evaluate recommendation models using Precision@k, Recall@k, and Coverage
from sklearn.metrics import precision_score, recall_score

def evaluate_recommendations(filtered_df, recommendations, processed_df, user_preferences, k=5):
    """
    Evaluate the recommendation model using Precision@k, Recall@k, and Coverage.

    Parameters:
    - filtered_df: DataFrame of filtered recipes matching user preferences.
    - recommendations: DataFrame of recommended recipes.
    - dataset: Full dataset of recipes.
    - user_preferences: Dictionary of user-specified preferences (e.g., {'cuisine': 'Indian'}).
    - k: Number of top recommendations to consider (default is 5).

    Returns:
    - metrics: Dictionary containing Precision@k, Recall@k, and Coverage.
    """
    # 1. Precision@k
    relevant_recipes = filtered_df  # Recipes that match all user preferences
    recommended_top_k = recommendations.head(k)  # Top-k recommendations

    # Check if the recommended recipes are in the relevant recipes
    relevant_in_top_k = recommended_top_k['name'].isin(relevant_recipes['name']).sum()
    precision_at_k = relevant_in_top_k / k

    # 2. Recall@k
    total_relevant = relevant_recipes.shape[0]
    recall_at_k = relevant_in_top_k / total_relevant if total_relevant > 0 else 0

    # 3. Coverage
    unique_recommended = recommendations['name'].nunique()
    total_items = processed_df['name'].nunique()
    coverage = unique_recommended / total_items

    metrics = {
        'Precision@k': round(precision_at_k, 4),
        'Recall@k': round(recall_at_k, 4),
        'Coverage': round(coverage, 4),
    }

    return metrics

# Example 
# user_preferences = {'cuisine': 'fusion', 'course': 'snack', 'diet': 'Vegetarian'}

# Find filtered recipes based on user preferences
# filtered_df = filter_recipes(processed_df, "masala dosa", **user_preferences)

# Get recommendations
# vectorizer = TfidfVectorizer()
# model = NMF
# recommendations = get_recommendations(filtered_df, vectorizer, model, num_recommendations=10)

# # Evaluate the recommendations
# metrics = evaluate_recommendations(filtered_df, recommendations, processed_df, user_preferences, k=5)
# print(metrics)