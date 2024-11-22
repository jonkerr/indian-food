import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


processed_df = pd.read_pickle("data/processed_recipes.pkl")


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


def compute_coherence_scores(filtered_df, vectorizer, model, min_topics=2, max_topics_limit = 10):
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


def get_recommendations(filtered_df, vectorizer, model, num_recommendations=5): 

    best_num_topics = compute_coherence_scores(filtered_df, vectorizer = vectorizer, model = model)

    vectorizer_matrix = vectorizer.fit_transform(filtered_df["combined_name_ingredients"])

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