import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error



processed_df = pd.read_pickle("data/processed_recipes.pkl")



# Define a function to filter recipes based on user preferences
def get_recommendations_nmf_tfidf(dish_name, num_recommendations=5, ingredients=None, cuisine=None, course=None, diet=None, prep_time=None, allergen_type=None): 

    # Filter recipes based on the provided criteria
    filtered_df = processed_df[processed_df['name'].str.contains(dish_name, case=False, na=False)]
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

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df['processed_name'])

    # Ensure all values in tfidf_matrix are non-negative and finite
    tfidf_matrix = np.nan_to_num(tfidf_matrix.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    # Dynamically determine the best number of topics
    min_topics = 2
    max_topics = min(15, tfidf_matrix.shape[0])  # Limit to the number of rows
    best_num_topics = min_topics
    best_error = float('inf')

    for num_topics in range(min_topics, max_topics + 1):
        nmf_model = NMF(n_components=num_topics, random_state=42)
        nmf_model.fit(tfidf_matrix)
        topic_matrix = nmf_model.transform(tfidf_matrix)
        reconstruction_error = mean_squared_error(tfidf_matrix, np.dot(topic_matrix, nmf_model.components_))
    
        if reconstruction_error < best_error:
            best_error = reconstruction_error
            best_num_topics = num_topics

    print(f"Best number of topics: {best_num_topics}")


    # Train the NMF model with the optimal number of topics    
    nmf_tfidf_model = NMF(n_components=best_num_topics, random_state=42)
    nmf_tfidf_model.fit(tfidf_matrix)
    topic_matrix = nmf_tfidf_model.transform(tfidf_matrix)  # Topic distribution for each recipe

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



# Define a function to filter recipes based on user preferences
def get_recommendations_nmf_count(dish_name, num_recommendations=5, ingredients=None, cuisine=None, course=None, diet=None, prep_time=None, allergen_type=None): 

    # Filter recipes based on the provided criteria
    filtered_df = processed_df[processed_df['name'].str.contains(dish_name, case=False, na=False)]
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

    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(filtered_df['processed_name'])

    # Ensure all values in tfidf_matrix are non-negative and finite
    count_matrix = np.nan_to_num(count_matrix.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    # Dynamically determine the best number of topics
    min_topics = 2
    max_topics = min(15, count_matrix.shape[0])  # Limit to the number of rows
    best_num_topics = min_topics
    best_error = float('inf')

    for num_topics in range(min_topics, max_topics + 1):
        nmf_model = NMF(n_components=num_topics, random_state=42)
        nmf_model.fit(count_matrix)
        topic_matrix = nmf_model.transform(count_matrix)
        reconstruction_error = mean_squared_error(count_matrix, np.dot(topic_matrix, nmf_model.components_))
    
        if reconstruction_error < best_error:
            best_error = reconstruction_error
            best_num_topics = num_topics

    print(f"Best number of topics: {best_num_topics}")


    # Train the NMF model with the optimal number of topics    
    nmf_count_model = NMF(n_components=best_num_topics, random_state=42)
    nmf_count_model.fit(count_matrix)
    topic_matrix = nmf_count_model.transform(count_matrix)  # Topic distribution for each recipe

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




# Define a function to filter recipes based on user preferences
def get_recommendations_svd_tfidf(dish_name, num_recommendations=5, ingredients=None, cuisine=None, course=None, diet=None, prep_time=None, allergen_type=None): 

    # Filter recipes based on the provided criteria
    filtered_df = processed_df[processed_df['name'].str.contains(dish_name, case=False, na=False)]
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

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df['processed_name'])

    # Ensure all values in tfidf_matrix are non-negative and finite
    tfidf_matrix = np.nan_to_num(tfidf_matrix.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    # Dynamically determine the best number of topics
    min_topics = 2
    max_topics = min(15, tfidf_matrix.shape[0])  # Limit to the number of rows
    best_num_topics = min_topics
    best_error = float('inf')

    for num_topics in range(min_topics, max_topics + 1):
        svd_model = TruncatedSVD(n_components=num_topics, random_state=42)
        svd_model.fit(tfidf_matrix)
        topic_matrix = svd_model.transform(tfidf_matrix)
        reconstruction_error = mean_squared_error(tfidf_matrix, np.dot(topic_matrix, svd_model.components_))
    
        if reconstruction_error < best_error:
            best_error = reconstruction_error
            best_num_topics = num_topics

    print(f"Best number of topics: {best_num_topics}")


    # Train the NMF model with the optimal number of topics    
    svd_tfidf_model = TruncatedSVD(n_components=best_num_topics, random_state=42)
    svd_tfidf_model.fit(tfidf_matrix)
    topic_matrix = svd_tfidf_model.transform(tfidf_matrix)  # Topic distribution for each recipe

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



# Define a function to filter recipes based on user preferences
def get_recommendations_svd_count(dish_name, num_recommendations=5, ingredients=None, cuisine=None, course=None, diet=None, prep_time=None, allergen_type=None): 

    # Filter recipes based on the provided criteria
    filtered_df = processed_df[processed_df['name'].str.contains(dish_name, case=False, na=False)]
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

    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(filtered_df['processed_name'])

    # Ensure all values in count_matrix are non-negative and finite
    count_matrix = np.nan_to_num(count_matrix.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    # Dynamically determine the best number of topics
    min_topics = 2
    max_topics = min(15, count_matrix.shape[0])  # Limit to the number of rows
    best_num_topics = min_topics
    best_error = float('inf')

    for num_topics in range(min_topics, max_topics + 1):
        svd_model = TruncatedSVD(n_components=num_topics, random_state=42)
        svd_model.fit(count_matrix)
        topic_matrix = svd_model.transform(count_matrix)
        reconstruction_error = mean_squared_error(count_matrix, np.dot(topic_matrix, svd_model.components_))
    
        if reconstruction_error < best_error:
            best_error = reconstruction_error
            best_num_topics = num_topics

    print(f"Best number of topics: {best_num_topics}")


    # Train the NMF model with the optimal number of topics    
    svd_count_model = TruncatedSVD(n_components=best_num_topics, random_state=42)
    svd_count_model.fit(count_matrix)
    topic_matrix = svd_count_model.transform(count_matrix)  # Topic distribution for each recipe

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