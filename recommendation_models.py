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
processed_df


# In[3]:


tfidf_vectorizer = TfidfVectorizer()
count_vectorizer = CountVectorizer()


# Find dish name in recipes and filter them based on user preferences

# In[43]:


def filter_recipes (df, dish_name, cuisine=None, course=None, diet=None, prep_time=None, allergen_type=None, debug=False): 
  
    filtered_df = df[df['name'].str.contains(dish_name, case=False, na=False)]
    print ((f"Number of found dishes before filtering : {filtered_df.shape[0]}"))

    if cuisine:
        filtered_df = filtered_df[filtered_df['cuisine'] .str.lower() == cuisine.lower()]
    if course:
        filtered_df = filtered_df[filtered_df['course'].str.lower() == course.lower()]
    if diet:
        filtered_df = filtered_df[filtered_df['diet'] .str.lower() == diet.lower()]
    if prep_time:
        filtered_df = filtered_df[filtered_df['categorized_prep_time'] == prep_time]
    if allergen_type:
        # Exclude recipes with allergens in user preference
        allergen_set = set(allergen_type)
        filtered_df = filtered_df[~filtered_df['allergen_type'].apply(lambda x: bool(set(x) & allergen_set))]

    if filtered_df.empty:
        print("No recipes found matching the criteria.")
        return None
    print(f"Number of dishes after filtering: {filtered_df.shape[0]}")
    return filtered_df


# In[87]:


filtered_df = filter_recipes (processed_df, "cabbage carrot")## diet = "Vegetarian")
filtered_df


# Calculate the best number of topics using coherence score for the filtered dataframe

# In[88]:


def compute_coherence_scores(filtered_df, vectorizer, model, min_topics=2, max_topics_limit = 10 ):
    coherence_scores = []
    
    # Transform texts to TF-IDF matrix
    vectorizer_matrix = vectorizer.fit_transform(filtered_df["combined_name_ingredients"])
    
    # Tokenize each text in `texts` for Gensim's coherence calculation
    tokenized_texts = [text.split() for text in filtered_df["combined_name_ingredients"]]
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    max_topics = min(max_topics_limit, vectorizer_matrix.shape[0])
    num_topics_range = range(min_topics, max_topics + 1)
        
    # Iterate over each specified topic count
    for num_topics in num_topics_range :
        called_model = model(n_components=num_topics, random_state=42)
        called_matrix = called_model.fit_transform(vectorizer_matrix)
        
        # Extract top words for each topic and format as list of tokens
        feature_names = vectorizer.get_feature_names_out()
        topics = [[feature_names[i] for i in topic.argsort()[:-11:-1]] for topic in called_model.components_]
        
        tokenized_topics = [[word for word in topic] for topic in topics]# Coherence model requires a list of tokenized topics

        coherence_model = CoherenceModel(
            topics=tokenized_topics, 
            texts=tokenized_texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
        
        #print(f"Number of Topics: {num_topics}, Coherence Score: {coherence_score}")

    # Find the best number of topics
    best_num_topics = num_topics_range[coherence_scores.index(max(coherence_scores))]
    print(f"Best Number of Topics: {best_num_topics} with Coherence Score: {max(coherence_scores)}")

    return best_num_topics


# Train the recommendation model using topic models and vectorizers

# In[89]:


def get_recommendations(filtered_df, vectorizer, model, num_recommendations=5): 

    best_num_topics = compute_coherence_scores(filtered_df, vectorizer, model)

    vectorizer_matrix = vectorizer.fit_transform(filtered_df['combined_name_ingredients'])

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

    # Find the index of the dish within the filtered DataFrame
    try:
        dish_index = filtered_df.reset_index().index[0] # Get the first match if there are multiple
    except IndexError:
        raise IndexError("Filtered DataFrame is empty. No recipes match the criteria.")
    
    cosine_similarities = cosine_similarity(topic_matrix)
    
    # Get indices of the most similar recipes
    similar_indices = cosine_similarities[dish_index].argsort()[-num_recommendations-1:-1][::-1]

    # Fetch recommended recipes
    recommended_recipes = filtered_df.iloc[similar_indices].copy()
    recommended_recipes["similarity_score"] = cosine_similarities[dish_index][similar_indices]

    # Scale and format similarity scores to percentages
    recommended_recipes['similarity_score'] = (recommended_recipes['similarity_score'] * 100).round(2).astype(str) + '%'

    return recommended_recipes[['name', 'similarity_score', 'cleaned_ingredients', 'cuisine', 'course', 'diet', 'allergens', 'prep_time']]


# Get recipe recommendation using the best NMF model on TF-IDF vectorizer

# In[38]:


recommended_recipes_nmf_tfidf = get_recommendations(filtered_df, tfidf_vectorizer, NMF) 
recommended_recipes_nmf_tfidf


# Get recipe recommendation using best NMF model on Count vectorizer

# In[9]:


recommended_recipes_nmf_count = get_recommendations(filtered_df, count_vectorizer, NMF) 
recommended_recipes_nmf_count


# Get recipe recommendation using best SVD model on TF-IDF vectorizer

# In[10]:


recommended_recipes_svd_tfidf = get_recommendations(filtered_df, tfidf_vectorizer, TruncatedSVD) 
recommended_recipes_svd_tfidf


# Get recipe recommendation using best SVD model on Count vectorizer

# In[11]:


recommended_recipes_svd_count = get_recommendations(filtered_df, count_vectorizer, TruncatedSVD) 
recommended_recipes_svd_count


# In[ ]:





# Calculate the best number of topics using coherence score for only filtered_df["processed_name"]

# In[12]:


def compute_coherence_scores_name(filtered_df, vectorizer, model, min_topics=2, max_topics_limit = 30):
    coherence_scores = []
    
    # Transform texts to TF-IDF matrix
    vectorizer_matrix = vectorizer.fit_transform(filtered_df["processed_name"])
    
    # Tokenize each text in `texts` for Gensim's coherence calculation
    tokenized_texts = [text.split() for text in filtered_df["processed_name"]]
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    max_topics = min(max_topics_limit, vectorizer_matrix.shape[0])
    num_topics_range = range(min_topics, max_topics + 1)
        
    # Iterate over each specified topic count
    for num_topics in num_topics_range :
        called_model = model(n_components=num_topics, random_state=42)
        called_matrix = called_model.fit_transform(vectorizer_matrix)
        
        # Extract top words for each topic and format as list of tokens
        feature_names = vectorizer.get_feature_names_out()
        topics = [[feature_names[i] for i in topic.argsort()[:-11:-1]] for topic in called_model.components_]
        
        tokenized_topics = [[word for word in topic] for topic in topics]# Coherence model requires a list of tokenized topics

        coherence_model = CoherenceModel(
            topics=tokenized_topics, 
            texts=tokenized_texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
        
        #print(f"Number of Topics: {num_topics}, Coherence Score: {coherence_score}")

    # Find the best number of topics
    best_num_topics = num_topics_range[coherence_scores.index(max(coherence_scores))]
    print(f"Best Number of Topics: {best_num_topics} with Coherence Score: {max(coherence_scores)}")

    return best_num_topics


# Train the recommendation model using topic models and vectorizers for only filtered_df["processed_name"]

# In[13]:


def get_recommendations_name(filtered_df, vectorizer, model, num_recommendations=5): 

    best_num_topics = compute_coherence_scores_name(filtered_df, vectorizer, model)

    vectorizer_matrix = vectorizer.fit_transform(filtered_df["processed_name"])

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


# Compare recipes recommended by NMF and SVD models on Tf-IDF and Count vectorizers using only filtered_df["processed_name"]

# In[14]:


name_recommended_recipes_nmf_tfidf = get_recommendations_name(filtered_df, tfidf_vectorizer, NMF) 
name_recommended_recipes_nmf_tfidf


# In[15]:


name_recommended_recipes_nmf_count = get_recommendations_name(filtered_df, count_vectorizer, NMF) 
name_recommended_recipes_nmf_count


# In[16]:


name_recommended_recipes_svd_tfidf = get_recommendations_name(filtered_df, tfidf_vectorizer, TruncatedSVD) 
name_recommended_recipes_svd_tfidf


# In[17]:


name_recommended_recipes_svd_count = get_recommendations_name(filtered_df, count_vectorizer, TruncatedSVD) 
name_recommended_recipes_svd_count


# In[ ]:





# In[18]:


def save_model(model, file_name, path='models'):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(model, f)


# In[19]:


def save_vectorizer(vectorizer, file_name, path='models'):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(vectorizer, f)


# In[ ]:


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


# In[ ]:


#tfidf_vect = get_recommendations_save_pickles(filtered_df, tfidf_vectorizer, NMF, "tfidf_vectorizer.pkl", path='models')


# In[ ]:


#count_vect = get_recommendations_save_pickles(filtered_df, count_vectorizer, NMF, "count_vectorizer.pkl", path='models')


# In[ ]:


#nmf_tfid_model = get_recommendations(filtered_df, tfidf_vectorizer, NMF, "nmf_tfidf_model.pkl", path='models')


# In[ ]:


#nmf_count_model = get_recommendations(filtered_df, count_vectorizer, NMF, "nmf_count_model.pkl", path='models')


# In[ ]:


#svd_tfidf_model = get_recommendations(filtered_df, tfidf_vectorizer, TruncatedSVD,"svd_tfidf_model.pkl", path='models')


# In[ ]:


#svd_count_model = get_recommendations(filtered_df, count_vectorizer,TruncatedSVD, "svd_count_model.pkl", path='models')#


# In[27]:


def get_model(file_name, path='models'):
     
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file '{file_name}' does not exist in the path '{path}'.")
    
    with open(full_path, 'rb') as f:
        return pickle.load(f)


# In[ ]:


#nmf_tfidf = get_model("nmf_tfidf_model.pkl", "models")
#nmf_tfidf 


# In[ ]:


#nmf_count = get_model("nmf_count_model.pkl", "models")
#nmf_count


# In[ ]:


#svd_tfidf = get_model("svd_tfidf_model.pkl", "models")
#svd_tfidf


# In[ ]:


#svd_count = get_model("svd_count_model.pkl", "models")
#svd_count


# In[77]:


def get_vectorizer(file_name, path='models'):
     
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file '{file_name}' does not exist in the path '{path}'.")
    
    with open(full_path, 'rb') as f:
        return pickle.load(f)


# In[ ]:


#tfidf_vec = get_vectorizer("tfidf_vectorizer.pkl", "models")
#tfidf_vec


# In[ ]:


#count_vec = get_vectorizer("count_vectorizer.pkl", "models")
#count_vec


# In[46]:


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

# Example usage
# Define user preferences
user_preferences = {
    'cuisine': 'fusion',
    'course': 'snack',
    'diet': 'Vegetarian',
}

# Find filtered recipes based on user preferences
filtered_df = filter_recipes(processed_df, "masala dosa", **user_preferences)

# Get recommendations
vectorizer = TfidfVectorizer()
model = NMF
recommendations = get_recommendations(filtered_df, vectorizer, model, num_recommendations=10)

# Evaluate the recommendations
metrics = evaluate_recommendations(filtered_df, recommendations, processed_df, user_preferences, k=5)
print(metrics)

