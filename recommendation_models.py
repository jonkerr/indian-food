import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


processed_df = pd.read_pickle("data/processed_recipes.pkl")


def find_filter_recipes (dish_name, df, ingredients=None, cuisine=None, course=None, diet=None, prep_time=None, allergen_type=None): 
  
    filtered_df = df[df['name'].str.contains(dish_name, case=False, na=False)]
    print ((f"Number of found dishes before filtering : {filtered_df.shape[0]}"))

    if ingredients:
        # Check if all specified ingredients are present in the cleaned_ingredients list
        filtered_df = filtered_df[filtered_df['cleaned_ingredients'].apply(lambda x: all(ing in x for ing in ingredients))]            
    if cuisine:
        filtered_df = filtered_df[filtered_df['cuisine'] == cuisine]
    if course:
        filtered_df = filtered_df[filtered_df['course'] == course]
    if diet:
        filtered_df = filtered_df[filtered_df['diet'] == diet]
    if prep_time:
        filtered_df = filtered_df[filtered_df['categorized_prep_time'] == prep_time]
    if allergen_type:
        # Exclude recipes with allergens in user preference
        filtered_df = filtered_df[~filtered_df['allergen_type'].apply(lambda x: bool(set(x) & set(allergen_type)))]

    if filtered_df.empty:
        print("No recipes found matching the criteria.")
        return None
    return filtered_df


def calculate_best_num_topics(filtered_df, vectorizer, model, min_topics=2, max_topics_limit=15):

    filtered_df = filtered_df.copy()
    vectorizer_matrix = vectorizer.fit_transform(filtered_df["processed_name"])

    # Ensure the matrix is valid
    vectorizer_matrix = np.nan_to_num(vectorizer_matrix.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    # Determine the maximum number of topics based on data size
    max_topics = min(max_topics_limit, vectorizer_matrix.shape[0])
    best_num_topics = min_topics
    best_error = float('inf')

    # Evaluate reconstruction error for each topic count
    for num_topics in range(min_topics, max_topics + 1):
        called_model = model(n_components=num_topics, random_state=42)
        called_model.fit(vectorizer_matrix)
        topic_matrix = called_model.transform(vectorizer_matrix)
        reconstruction_error = mean_squared_error(vectorizer_matrix, np.dot(topic_matrix, called_model.components_))
        
        # Update best error and topic count
        if reconstruction_error < best_error:
            best_error = reconstruction_error
            best_num_topics = num_topics

    print(f"Best number of topics: {best_num_topics}")
    return best_num_topics


def get_recommendations(filtered_df, vectorizer, model, num_recommendations=5): 

    best_num_topics = calculate_best_num_topics(filtered_df, vectorizer = vectorizer, model = model)

    vectorizer_matrix = vectorizer.fit_transform(filtered_df['processed_name'])

    # Ensure all values in tfidf_matrix are non-negative and finite
    vectorizer_matrix = np.nan_to_num(vectorizer_matrix.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    # Train the NMF model with the optimal number of topics    
    called_model = model(n_components=best_num_topics, random_state=42)
    called_model.fit(vectorizer_matrix)
    topic_matrix = called_model.transform(vectorizer_matrix)  # Topic distribution for each recipe

    filtered_df = filtered_df.copy()

    # Add topic distributions to the filtered DataFrame
    for i in range(best_num_topics):
        filtered_df[f'topic_{i}'] = topic_matrix[:, i]

    # Locate the index of the dish within the filtered DataFrame
    dish_index = filtered_df.reset_index().index[0]  # Get the first match if there are multiple

    # Calculate cosine similarity for the topics
    cosine_similarities = cosine_similarity(topic_matrix)
    
    # Get indices of the most similar recipes
    similar_indices = cosine_similarities[dish_index].argsort()[-num_recommendations-1:-1][::-1]

    # Fetch recommended recipes
    recommended_recipes = filtered_df.iloc[similar_indices].copy()
    recommended_recipes["similarity_score"] = cosine_similarities[dish_index][similar_indices]

    # Scale and format similarity scores to percentages
    recommended_recipes['similarity_score'] = (recommended_recipes['similarity_score'] * 100).round(2).astype(str) + '%'

    return recommended_recipes[['name', 'similarity_score', 'cleaned_ingredients', 'cuisine', 'course', 'diet', 'allergens', 'prep_time']]


def get_recommendations_nmf_tfidf(filtered_df): 
  return get_recommendations(filtered_df, TfidfVectorizer(), NMF, num_recommendations=5)

def get_recommendations_nmf_count(filtered_df): 
  return get_recommendations(filtered_df, CountVectorizer(), NMF, num_recommendations=5)

def get_recommendations_svd_tfidf(filtered_df):
  return get_recommendations(filtered_df, TfidfVectorizer(), TruncatedSVD, num_recommendations=5)

def get_recommendations_svd_count(filtered_df):
  return get_recommendations(filtered_df, CountVectorizer(), TruncatedSVD, num_recommendations=5)